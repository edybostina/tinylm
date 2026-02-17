#include "scheduler.h"

uint64_t Scheduler::add_request(const std::string& user_prompt, const std::vector<int>& token_ids) {
	uint64_t req_id = next_request_id_++;
	Sequence new_seq(req_id, user_prompt, token_ids, block_size_);

	waiting_sequences_.push_back(std::move(new_seq));
	return req_id;
}

std::vector<StepResult> Scheduler::step() {
	std::vector<StepResult> results;

	while (!waiting_sequences_.empty()) {
		Sequence& seq = waiting_sequences_.front();

		int blocks_needed = (seq.get_seq_len() + block_size_ - 1) / block_size_;
		int blocks_owned = seq.get_block_table().size();
		int blocks_to_allocate = blocks_needed - blocks_owned;

		if (kv_cache_->available_blocks() < blocks_to_allocate) {
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

	for (Sequence& seq : running_sequences_) {
		if (seq.get_token_ids().size() == seq.get_prompt_length() && seq.get_token_ids().size() > 1) {
			for (int id : seq.get_token_ids()) {
				forward_pass_tokens.push_back(id);
			}
			tokens_per_sequence.push_back(seq.get_prompt_length());
		} else {
			forward_pass_tokens.push_back(seq.get_last_token_id());
			tokens_per_sequence.push_back(1);
		}
	}

	torch::Tensor input_tensor =
		torch::from_blob(forward_pass_tokens.data(), {1, static_cast<long long>(forward_pass_tokens.size())},
						 torch::TensorOptions().dtype(torch::kInt64))
			.to(model_loader_.get_device())
			.clone();

	std::vector<torch::IValue> inputs = {input_tensor};
	torch::Tensor logits = model_loader_.forward(inputs).toTensor();

	int current_token_offset = 0;
	int seq_index = 0;

	for (Sequence& seq : running_sequences_) {
		int num_tokens_for_this_seq = tokens_per_sequence[seq_index];
		int target_logit_index = current_token_offset + num_tokens_for_this_seq - 1;

		torch::Tensor seq_logits = logits[0][target_logit_index];

		int new_token_id = torch::argmax(seq_logits, -1).item<int>();

		bool is_finished = (new_token_id == tokenizer_.get_eos_id());

		if (seq.needs_new_block()) {
			if (kv_cache_->available_blocks() > 0) {
				seq.add_block(kv_cache_->allocate_block());
			} else {
				throw std::runtime_error("OOM during generation!");
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
					   [](const Sequence& seq) { return seq.get_state() == SequenceState::FINISHED; }),
		running_sequences_.end());

	return results;
}
