#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <thread>
#include <chrono>

#include "model/tokenizer.h"
#include "model/loader.h"
#include "model/kv_cache.h"
#include "scheduler/scheduler.h"

int main() {
	try {
		torch::Device device(torch::kCPU);

		Tokenizer tokenizer("sentencepiece_models/sentencepiece.bpe.model");
		Loader loader("pytorch_weights/tiny.pt", device);

		int block_size = 16;
		auto kv_cache = std::make_unique<PagedKVCache>(100, 2, 64, block_size, device, torch::kFloat32);

		Scheduler scheduler(std::move(kv_cache), loader, tokenizer, block_size);

		std::vector<std::string> prompts = {"Hello, my name is tinylm and", "The capital of France is"};

		for (const auto& prompt : prompts) {
			std::vector<int> tokens = tokenizer.encode(prompt, true);
			uint64_t req_id = scheduler.add_request(prompt, tokens);
			std::cout << "[Server] Added Request " << req_id << " (" << tokens.size() << " prompt tokens)\n";
		}

		int step_count = 0;
		bool active = true;

		while (active) {
			std::vector<StepResult> results = scheduler.step();

			if (results.empty()) {
				active = false;
				break;
			}

			for (const auto& res : results) {
				std::string decoded_text = tokenizer.decode(res.new_token_id);

				std::cout << "[Req " << res.request_id << "] generated: '" << decoded_text
						  << "' (ID: " << res.new_token_id << ")\n";

				if (res.is_finished) {
					std::cout << "[Req " << res.request_id << "] --- FINISHED ---\n";
				}
			}

			step_count++;

			if (step_count >= 20) {
				std::cout << "\n[Server] Reached 20 max steps for simulation. Stopping.\n";
				break;
			}

			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		}

	} catch (const std::exception& e) {
		std::cerr << "Fatal Error: " << e.what() << "\n";
	}

	return 0;
}
