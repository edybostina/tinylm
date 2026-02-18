#include "scheduler.h"
#include <limits>
#include <c10/util/Exception.h>
#include "../logger.h"

static int sample_token(torch::Tensor logits, float temperature, int top_k, float top_p) {
	if (temperature <= 0.0f) {
		return torch::argmax(logits, -1).item<int>();
	}

	logits = logits / temperature;

	if (top_k > 0) {
		auto [topk_values, topk_indices] = torch::topk(logits, std::min(top_k, (int)logits.size(0)));
		torch::Tensor mask = torch::full_like(logits, -std::numeric_limits<float>::infinity());
		mask.scatter_(0, topk_indices, topk_values);
		logits = mask;
	}

	if (top_p < 1.0f) {
		torch::Tensor sorted_logits, sorted_indices;
		std::tie(sorted_logits, sorted_indices) = torch::sort(logits, /*dim=*/0, /*descending=*/true);

		torch::Tensor probs = torch::softmax(sorted_logits, 0);
		torch::Tensor cumulative_probs = torch::cumsum(probs, 0);

		torch::Tensor remove_mask = cumulative_probs - probs > top_p;

		sorted_logits.masked_fill_(remove_mask, -std::numeric_limits<float>::infinity());

		torch::Tensor original_order_logits = torch::zeros_like(logits);
		original_order_logits.scatter_(0, sorted_indices, sorted_logits);
		logits = original_order_logits;
	}

	torch::Tensor probs = torch::softmax(logits, 0);

	if (probs.isnan().any().item<bool>()) {
		LOG_WARN("Scheduler") << "NaN detected in probabilities, falling back to argmax";
		return torch::argmax(logits, -1).item<int>();
	}

	return torch::multinomial(probs, 1).item<int>();
}

void Scheduler::add_request(uint64_t request_id, const std::string &user_prompt, const std::vector<int> &token_ids,
							int max_tokens, float temperature, int top_k, float top_p) {
	Sequence new_seq(request_id, user_prompt, token_ids, max_tokens, temperature, top_k, top_p);
	waiting_sequences_.push_back(std::move(new_seq));
}

std::vector<StepResult> Scheduler::step() {
	std::vector<StepResult> results;

	while (!waiting_sequences_.empty() && static_cast<int>(running_sequences_.size()) < max_batch_size_) {
		Sequence &seq = waiting_sequences_.front();
		seq.set_state(SequenceState::RUNNING);
		running_sequences_.push_back(std::move(seq));
		waiting_sequences_.pop_front();
	}

	if (running_sequences_.empty()) {
		return results;
	}

	auto device = prefill_loader_.get_device();

	for (Sequence &seq : running_sequences_) {
		torch::Tensor logits;

		if (!seq.is_prefilled()) {
			const auto &tokens = seq.get_token_ids();
			std::vector<int64_t> token_ids_64(tokens.begin(), tokens.end());

			auto input = torch::from_blob(token_ids_64.data(), {1, (int64_t)token_ids_64.size()}, torch::kInt64)
							 .to(device)
							 .clone();

			auto output = prefill_loader_.forward({input});
			auto tuple = output.toTuple();
			auto all_logits = tuple->elements()[0].toTensor();	// [1, seq_len, vocab]
			auto keys = tuple->elements()[1].toTensor();  // [layers, 1, heads, seq_len, dim]
			auto values = tuple->elements()[2].toTensor();

			seq.set_kv_cache(keys, values);
			seq.set_prefilled(true);

			logits = all_logits[0][-1];	 // [vocab]

		} else {
			int64_t last_token = seq.get_last_token_id();
			auto input = torch::tensor({{last_token}}, torch::kInt64).to(device);

			auto output = decode_loader_.forward({input, seq.get_key_cache(), seq.get_value_cache()});
			auto tuple = output.toTuple();
			auto all_logits = tuple->elements()[0].toTensor();	// [1, 1, vocab]
			auto new_keys = tuple->elements()[1].toTensor();
			auto new_values = tuple->elements()[2].toTensor();

			seq.set_kv_cache(new_keys, new_values);

			logits = all_logits[0][-1];	 // [vocab]
		}

		int new_token_id = sample_token(logits, seq.get_temperature(), seq.get_top_k(), seq.get_top_p());

		bool is_finished = (new_token_id == tokenizer_.get_eos_id());
		int tokens_generated = seq.get_token_ids().size() - seq.get_prompt_length();
		if (tokens_generated >= seq.get_max_tokens())
			is_finished = true;

		seq.append_token(new_token_id);
		if (is_finished) {
			seq.set_state(SequenceState::FINISHED);
		}
		results.push_back({seq.get_id(), new_token_id, is_finished});
	}

	running_sequences_.erase(
		std::remove_if(running_sequences_.begin(), running_sequences_.end(),
					   [](const Sequence &seq) { return seq.get_state() == SequenceState::FINISHED; }),
		running_sequences_.end());

	return results;
}
