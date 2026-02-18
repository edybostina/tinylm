#pragma once

#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <mutex>
#include <atomic>

#include "thread_safe_queue.h"
#include "../scheduler/scheduler.h"

struct InboundRequest {
	uint64_t request_id;
	std::string prompt;
	std::vector<int> token_ids;
	int max_tokens;
	float temperature;
	int top_k;
	float top_p;
};

class Coordinator {
   public:
	ThreadSafeQueue<InboundRequest> request_queue;
	std::atomic<bool> stop_requested{false};

	uint64_t next_id() {
		return global_request_id++;
	}

	std::shared_ptr<ThreadSafeQueue<StepResult>> register_queue(uint64_t req_id) {
		auto q = std::make_shared<ThreadSafeQueue<StepResult>>();
		std::lock_guard<std::mutex> lock(map_mutex_);
		response_queues_[req_id] = q;
		return q;
	}

	std::shared_ptr<ThreadSafeQueue<StepResult>> get_queue(uint64_t req_id) {
		std::lock_guard<std::mutex> lock(map_mutex_);
		auto it = response_queues_.find(req_id);
		return it != response_queues_.end() ? it->second : nullptr;
	}

	void remove_queue(uint64_t req_id) {
		std::lock_guard<std::mutex> lock(map_mutex_);
		response_queues_.erase(req_id);
		cancelled_requests_.erase(req_id);
	}

	void cancel_request(uint64_t req_id) {
		std::lock_guard<std::mutex> lock(map_mutex_);
		cancelled_requests_.insert(req_id);
	}

	bool is_cancelled(uint64_t req_id) {
		std::lock_guard<std::mutex> lock(map_mutex_);
		return cancelled_requests_.count(req_id) > 0;
	}

   private:
	std::mutex map_mutex_;
	std::atomic<uint64_t> global_request_id{0};
	std::unordered_map<uint64_t, std::shared_ptr<ThreadSafeQueue<StepResult>>> response_queues_;
	std::unordered_set<uint64_t> cancelled_requests_;
};
