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
		torch::Tensor cumulative_probs = torch::cumsum(torch::softmax(sorted_logits, 0), 0);
		torch::Tensor remove_mask = cumulative_probs - torch::softmax(sorted_logits, 0) > top_p;
		sorted_logits.masked_fill_(remove_mask, -std::numeric_limits<float>::infinity());
		torch::Tensor original_order_logits = torch::zeros_like(logits);
		original_order_logits.scatter_(0, sorted_indices, sorted_logits);
		logits = original_order_logits;
	}

	torch::Tensor probs = torch::softmax(logits, 0);
	return torch::multinomial(probs, 1).item<int>();
}

void Scheduler::add_request(uint64_t request_id, const std::string &user_prompt, const std::vector<int> &token_ids,
							int max_tokens, float temperature, int top_k, float top_p) {
	Sequence new_seq(request_id, user_prompt, token_ids, block_size_, max_tokens, temperature, top_k, top_p);
	waiting_sequences_.push_back(std::move(new_seq));
}

std::vector<StepResult> Scheduler::step() {
	std::vector<StepResult> results;

	while (!waiting_sequences_.empty() && static_cast<int>(running_sequences_.size()) < max_batch_size_) {
		Sequence &seq = waiting_sequences_.front();

		int blocks_needed = (seq.get_seq_len() + block_size_ - 1) / block_size_;
		int blocks_owned = seq.get_block_table().size();
		int blocks_to_allocate = blocks_needed - blocks_owned;

		if (kv_cache_->available_blocks() < blocks_to_allocate) {
			if (!running_sequences_.empty()) {
				Sequence &victim = running_sequences_.back();
				LOG_WARN("Scheduler") << "OOM: preempting sequence " << victim.get_id() << " to free "
									  << victim.get_block_table().size() << " blocks";
				kv_cache_->free_blocks(victim.get_block_table());
				victim.clear_blocks();
				victim.reset_to_prompt();
				victim.set_state(SequenceState::WAITING);
				waiting_sequences_.push_front(std::move(victim));
				running_sequences_.pop_back();
			}
			break;
		}

		for (int i = 0; i < blocks_to_allocate; ++i) {
			seq.add_block(kv_cache_->allocate_block());
		}

		seq.set_state(SequenceState::RUNNING);
		running_sequences_.push_back(std::move(seq));
		waiting_sequences_.pop_front();
	}

	if (running_sequences_.empty()) {
		return results;
	}

	std::vector<int64_t> forward_pass_tokens;
	std::vector<int> tokens_per_sequence;
	std::vector<int64_t> position_ids_vec;
	std::vector<int64_t> block_tables_vec;
	std::vector<int> slots_vec;

	int max_blocks_per_seq = 0;
	if (!running_sequences_.empty()) {
		for (const auto &seq : running_sequences_) {
			max_blocks_per_seq = std::max(max_blocks_per_seq, (int)seq.get_block_table().size());
		}
	}

	int batch_size = running_sequences_.size();
	torch::Tensor block_tables_tensor = torch::full({batch_size, max_blocks_per_seq}, -1, torch::kInt64);

	int seq_idx = 0;
	for (Sequence &seq : running_sequences_) {
		const auto &blocks = seq.get_block_table();
		for (size_t i = 0; i < blocks.size(); ++i) {
			block_tables_tensor[seq_idx][i] = blocks[i];
		}

		if (seq.get_token_ids().size() == seq.get_prompt_length() && seq.get_token_ids().size() > 1) {
			for (size_t i = 0; i < seq.get_token_ids().size(); ++i) {
				forward_pass_tokens.push_back(seq.get_token_ids()[i]);
				position_ids_vec.push_back(i);
				int64_t block_idx = i / block_size_;
				int64_t block_offset = i % block_size_;
				slots_vec.push_back(blocks[block_idx] * block_size_ + block_offset);
			}
			tokens_per_sequence.push_back(seq.get_prompt_length());
		} else {
			size_t pos = seq.get_token_ids().size() - 1;
			forward_pass_tokens.push_back(seq.get_last_token_id());
			position_ids_vec.push_back(pos);
			int64_t block_idx = pos / block_size_;
			int64_t block_offset = pos % block_size_;
			slots_vec.push_back(blocks[block_idx] * block_size_ + block_offset);
			tokens_per_sequence.push_back(1);
		}
		seq_idx++;
	}

	torch::Tensor input_tensor =
		torch::from_blob(forward_pass_tokens.data(), {1, static_cast<long long>(forward_pass_tokens.size())},
						 torch::TensorOptions().dtype(torch::kInt64))
			.to(model_loader_.get_device())
			.clone();

	torch::Tensor position_ids =
		torch::from_blob(position_ids_vec.data(), {1, static_cast<long long>(position_ids_vec.size())},
						 torch::TensorOptions().dtype(torch::kInt64))
			.to(model_loader_.get_device())
			.clone();

	block_tables_tensor = block_tables_tensor.to(model_loader_.get_device());

	torch::Tensor slots_tensor = torch::from_blob(slots_vec.data(), {static_cast<long long>(slots_vec.size())},
												  torch::TensorOptions().dtype(torch::kInt64))
									 .to(model_loader_.get_device())
									 .clone();

	torch::Tensor logits;
	try {
		std::vector<torch::IValue> inputs = {
			input_tensor,		 position_ids, kv_cache_->get_key_cache(), kv_cache_->get_value_cache(),
			block_tables_tensor, slots_tensor};
		logits = model_loader_.forward(inputs).toTensor();
	} catch (const c10::Error &e) {
		LOG_WARN("Scheduler") << "PagedAttention forward failed (" << e.msg() << "), falling back to naive forward.";
		std::vector<torch::IValue> inputs = {input_tensor};
		logits = model_loader_.forward(inputs).toTensor();
	}

	int current_token_offset = 0;
	int seq_index = 0;

	for (Sequence &seq : running_sequences_) {
		int num_tokens_for_this_seq = tokens_per_sequence[seq_index];
		int target_logit_index = current_token_offset + num_tokens_for_this_seq - 1;

		torch::Tensor seq_logits = logits[0][target_logit_index];

		int new_token_id = sample_token(seq_logits, seq.get_temperature(), seq.get_top_k(), seq.get_top_p());

		bool is_finished = (new_token_id == tokenizer_.get_eos_id());
		int tokens_generated = seq.get_token_ids().size() - seq.get_prompt_length();
		if (tokens_generated >= seq.get_max_tokens()) {
			is_finished = true;
		}

		if (seq.needs_new_block()) {
			if (kv_cache_->available_blocks() > 0) {
				seq.add_block(kv_cache_->allocate_block());
			} else {
				LOG_ERROR("Scheduler") << "OOM mid-generation for sequence " << seq.get_id()
									   << ", terminating gracefully.";
				seq.set_state(SequenceState::FINISHED);
				kv_cache_->free_blocks(seq.get_block_table());
				results.push_back({seq.get_id(), tokenizer_.get_eos_id(), true});
				current_token_offset += num_tokens_for_this_seq;
				seq_index++;
				continue;
			}
		}

		seq.append_token(new_token_id);

		if (is_finished) {
			seq.set_state(SequenceState::FINISHED);
			kv_cache_->free_blocks(seq.get_block_table());
		}

		results.push_back({seq.get_id(), new_token_id, is_finished});

		current_token_offset += num_tokens_for_this_seq;
		seq_index++;
	}

	running_sequences_.erase(
		std::remove_if(running_sequences_.begin(), running_sequences_.end(),
					   [](const Sequence &seq) { return seq.get_state() == SequenceState::FINISHED; }),
		running_sequences_.end());

	return results;
}
