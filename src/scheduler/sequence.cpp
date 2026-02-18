#include "sequence.h"

Sequence::Sequence(uint64_t request_id, const std::string &user_prompt, const std::vector<int> &prompt_tokens,
				   int max_tokens, float temperature, int top_k, float top_p)
	: request_id_(request_id),
	  user_prompt_(user_prompt),
	  token_ids_(prompt_tokens),
	  prompt_length_(prompt_tokens.size()),
	  max_tokens_(max_tokens),
	  temperature_(temperature),
	  top_k_(top_k),
	  top_p_(top_p),
	  state_(SequenceState::WAITING) {}

void Sequence::append_token(int token_id) {
	token_ids_.push_back(token_id);
}
