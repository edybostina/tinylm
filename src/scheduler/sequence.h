#pragma once

#include <string>
#include <vector>
#include <cstdint>

enum class SequenceState { WAITING, RUNNING, FINISHED };

class Sequence {
   private:
	uint64_t request_id_;
	std::string user_prompt_;

	std::vector<int> token_ids_;
	size_t prompt_length_;

	std::vector<int64_t> block_table_;
	int block_size_;

	SequenceState state_;

	int max_tokens_;

	float temperature_;
	int top_k_;
	float top_p_;

   public:
	Sequence(uint64_t request_id, const std::string &user_prompt, const std::vector<int> &prompt_tokens, int block_size,
			 int max_tokens, float temperature = 1.0f, int top_k = 0, float top_p = 1.0f);

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

	bool needs_new_block() const;
	void add_block(int64_t physical_block_index);
	void clear_blocks() {
		block_table_.clear();
	}
	
	void reset_to_prompt() {
		token_ids_.resize(prompt_length_);
	}
	const std::vector<int64_t> &get_block_table() const {
		return block_table_;
	}
};
