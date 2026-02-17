#include "sequence.h"

Sequence::Sequence(uint64_t request_id, const std::string& user_prompt, const std::vector<int>& prompt_tokens,
				   int block_size)
	: request_id_(request_id),
	  user_prompt_(user_prompt),
	  token_ids_(prompt_tokens),
	  prompt_length_(prompt_tokens.size()),
	  block_size_(block_size),
	  state_(SequenceState::WAITING) {}

void Sequence::append_token(int token_id) {
	token_ids_.push_back(token_id);
}

bool Sequence::needs_new_block() const {
	size_t current_capacity = block_table_.size() * block_size_;
	return token_ids_.size() >= current_capacity;
}

void Sequence::add_block(int64_t physical_block_index) {
	block_table_.push_back(physical_block_index);
}
