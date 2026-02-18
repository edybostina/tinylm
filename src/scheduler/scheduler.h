#pragma once

#include <torch/script.h>
#include <deque>
#include <vector>
#include <memory>
#include <string>

#include "../model/loader.h"
#include "../model/tokenizer.h"
#include "sequence.h"

struct StepResult {
	uint64_t request_id;
	int new_token_id;
	bool is_finished;
};

class Scheduler {
   private:
	Loader &prefill_loader_;
	Loader &decode_loader_;
	Tokenizer &tokenizer_;

	std::deque<Sequence> waiting_sequences_;
	std::vector<Sequence> running_sequences_;

	int max_batch_size_;

   public:
	Scheduler(Loader &prefill_loader, Loader &decode_loader, Tokenizer &tokenizer, int max_batch_size = 32)
		: prefill_loader_(prefill_loader),
		  decode_loader_(decode_loader),
		  tokenizer_(tokenizer),
		  max_batch_size_(max_batch_size) {}

	void add_request(uint64_t request_id, const std::string &user_prompt, const std::vector<int> &token_ids,
					 int max_tokens, float temperature = 1.0f, int top_k = 0, float top_p = 1.0f);
	std::vector<StepResult> step();

	bool has_work() const {
		return !waiting_sequences_.empty() || !running_sequences_.empty();
	}
};
