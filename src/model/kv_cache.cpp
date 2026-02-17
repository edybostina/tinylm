#include "kv_cache.h"

PagedKVCache::PagedKVCache(int64_t num_blocks, int64_t num_heads, int64_t head_dim, int64_t block_size,
						   torch::Device device, torch::Dtype dtype)
	: num_blocks_(num_blocks), num_heads_(num_heads), head_dim_(head_dim), block_size_(block_size) {
	keys_ = torch::empty({num_blocks_, num_heads_, block_size_, head_dim_},
						 torch::TensorOptions().device(device).dtype(dtype));
	values_ = torch::empty({num_blocks_, num_heads_, block_size_, head_dim_},
						   torch::TensorOptions().device(device).dtype(dtype));

	free_blocks_.reserve(num_blocks_);
	for (int64_t i = 0; i < num_blocks_; ++i) {
		free_blocks_.push_back(i);
	}
}

int64_t PagedKVCache::allocate_block() {
	if (free_blocks_.empty()) {
		throw std::runtime_error("No free blocks available in KV cache");
	}
	int64_t block_index = free_blocks_.back();
	free_blocks_.pop_back();
	return block_index;
}

void PagedKVCache::free_block(int64_t block_index) {
	if (block_index < 0 || block_index >= num_blocks_) {
		throw std::runtime_error("Invalid block index to free");
	}
	free_blocks_.push_back(block_index);
}

void PagedKVCache::free_blocks(const std::vector<int64_t>& block_indices) {
	for (int64_t block_index : block_indices) {
		free_block(block_index);
	}
}

int64_t PagedKVCache::available_blocks() const {
	return static_cast<int64_t>(free_blocks_.size());
}
