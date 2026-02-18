#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>

template <typename T>
class ThreadSafeQueue {
   private:
	std::queue<T> queue_;
	mutable std::mutex mutex_;
	std::condition_variable cond_var_;
	bool shutdown_ = false;

   public:
	ThreadSafeQueue() = default;

	void push(T value) {
		std::lock_guard<std::mutex> lock(mutex_);
		queue_.push(std::move(value));
		cond_var_.notify_one();
	}

	void shutdown() {
		std::lock_guard<std::mutex> lock(mutex_);
		shutdown_ = true;
		cond_var_.notify_all();
	}

	std::optional<T> wait_and_pop() {
		std::unique_lock<std::mutex> lock(mutex_);
		cond_var_.wait(lock, [this] { return !queue_.empty() || shutdown_; });
		if (queue_.empty()) {
			return std::nullopt;
		}
		T value = std::move(queue_.front());
		queue_.pop();
		return value;
	}

	std::optional<T> try_pop() {
		std::lock_guard<std::mutex> lock(mutex_);
		if (queue_.empty()) {
			return std::nullopt;
		}

		T value = std::move(queue_.front());
		queue_.pop();
		return value;
	}

	bool empty() const {
		std::lock_guard<std::mutex> lock(mutex_);
		return queue_.empty();
	}
};
