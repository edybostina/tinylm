#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <torch/script.h>

enum class SequenceState { WAITING, RUNNING, FINISHED };

class Sequence {
   private:
	uint64_t request_id_;
	std::string user_prompt_;

	std::vector<int> token_ids_;
	size_t prompt_length_;

	SequenceState state_;

	int max_tokens_;

	float temperature_;
	int top_k_;
	float top_p_;

	torch::Tensor key_cache_;
	torch::Tensor value_cache_;
	bool is_prefilled_ = false;

   public:
	Sequence(uint64_t request_id, const std::string &user_prompt, const std::vector<int> &prompt_tokens, int max_tokens,
			 float temperature = 1.0f, int top_k = 0, float top_p = 1.0f);

	uint64_t get_id() const {
		return request_id_;
	}

	int get_max_tokens() const {
		return max_tokens_;
	}

	float get_temperature() const {
		return temperature_;
	}
	int get_top_k() const {
		return top_k_;
	}
	float get_top_p() const {
		return top_p_;
	}

	size_t get_prompt_length() const {
		return prompt_length_;
	}
	SequenceState get_state() const {
		return state_;
	}
	void set_state(SequenceState state) {
		state_ = state;
	}

	void append_token(int token_id);
	const std::vector<int> &get_token_ids() const {
		return token_ids_;
	}
	int get_last_token_id() const {
		return token_ids_.back();
	}
	size_t get_seq_len() const {
		return token_ids_.size();
	}

	void reset_to_prompt() {
		token_ids_.resize(prompt_length_);
		key_cache_.reset();
		value_cache_.reset();
		is_prefilled_ = false;
	}

	// KV cache accessors
	bool is_prefilled() const {
		return is_prefilled_;
	}
	void set_prefilled(bool v) {
		is_prefilled_ = v;
	}

	const torch::Tensor &get_key_cache() const {
		return key_cache_;
	}
	const torch::Tensor &get_value_cache() const {
		return value_cache_;
	}
	void set_kv_cache(torch::Tensor keys, torch::Tensor values) {
		key_cache_ = std::move(keys);
		value_cache_ = std::move(values);
	}
};
