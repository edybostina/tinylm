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
	Loader& model_loader_;
	Tokenizer& tokenizer_;

	std::deque<Sequence> waiting_sequences_;
	std::vector<Sequence> running_sequences_;

	int block_size_;
	uint64_t next_request_id_ = 0;

   public:
	Scheduler(std::unique_ptr<PagedKVCache> kv_cache, Loader& loader, Tokenizer& tokenizer, int block_size)
		: kv_cache_(std::move(kv_cache)), model_loader_(loader), tokenizer_(tokenizer), block_size_(block_size) {}

	uint64_t add_request(const std::string& user_prompt, const std::vector<int>& token_ids);

	std::vector<StepResult> step();
};
