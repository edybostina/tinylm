#pragma once

#include <torch/script.h>
#include <deque>
#include <vector>
#include <memory>
#include <string>

#include "../model/loader.h"
#include "../model/kv_cache.h"
#include "../model/tokenizer.h"
#include "sequence.h"

struct StepResult {
	uint64_t request_id;
	int new_token_id;
	bool is_finished;
};

class Scheduler {
   private:
	std::unique_ptr<PagedKVCache> kv_cache_;
	Loader &model_loader_;
	Tokenizer &tokenizer_;

	std::deque<Sequence> waiting_sequences_;
	std::vector<Sequence> running_sequences_;

	int block_size_;
	int max_batch_size_;

   public:
	Scheduler(std::unique_ptr<PagedKVCache> kv_cache, Loader &loader, Tokenizer &tokenizer, int block_size,
			  int max_batch_size = 32)
		: kv_cache_(std::move(kv_cache)),
		  model_loader_(loader),
		  tokenizer_(tokenizer),
		  block_size_(block_size),
		  max_batch_size_(max_batch_size) {}

	void add_request(uint64_t request_id, const std::string &user_prompt, const std::vector<int> &token_ids,
					 int max_tokens, float temperature = 1.0f, int top_k = 0, float top_p = 1.0f);
	std::vector<StepResult> step();

	bool has_work() const {
		return !waiting_sequences_.empty() || !running_sequences_.empty();
	}
};
