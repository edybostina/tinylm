#pragma once

#include <torch/script.h>
#include <vector>
#include <stdexcept>

class PagedKVCache {
   private:
	torch::Tensor keys_;
	torch::Tensor values_;

	int64_t num_blocks_;
	int64_t num_heads_;
	int64_t head_dim_;
	int64_t block_size_;

	std::vector<int64_t> free_blocks_;

   public:
	PagedKVCache(int64_t num_blocks, int64_t num_heads, int64_t head_dim, int64_t block_size, torch::Device device,
				 torch::Dtype dtype);

	int64_t allocate_block();
	void free_block(int64_t block_index);
	void free_blocks(const std::vector<int64_t>& block_indices);

	int64_t available_blocks() const;

	const torch::Tensor& get_key_cache() const {
		return keys_;
	}
	const torch::Tensor& get_value_cache() const {
		return values_;
	}
};
