#include <iostream>
#include <memory>
#include <thread>
#include <chrono>
#include <grpcpp/grpcpp.h>

#include "model/loader.h"
#include "model/tokenizer.h"
#include "scheduler/scheduler.h"
#include "server/coordinator.h"
#include "server/service.h"
#include "config.h"
#include "logger.h"

void RunEngineLoop(std::shared_ptr<Scheduler> scheduler, std::shared_ptr<Coordinator> coordinator,
				   const Config &config) {
	LOG_INFO("Engine") << "Background thread started.";

	while (!coordinator->stop_requested) {
		if (!scheduler->has_work()) {
			auto req_opt = coordinator->request_queue.wait_and_pop();
			if (!req_opt)
				break;
			auto &req = *req_opt;
			LOG_INFO("Engine") << "Picked up request " << req.request_id;
			int max_gen = req.max_tokens > 0 ? req.max_tokens : config.default_max_tokens;
			float temp = req.temperature > 0.0f ? req.temperature : config.default_temperature;
			int top_k = req.top_k > 0 ? req.top_k : config.default_top_k;
			float top_p = req.top_p > 0.0f && req.top_p < 1.0f ? req.top_p : config.default_top_p;
			scheduler->add_request(req.request_id, req.prompt, req.token_ids, max_gen, temp, top_k, top_p);
		}

		while (auto req_opt = coordinator->request_queue.try_pop()) {
			auto &req = *req_opt;
			LOG_INFO("Engine") << "Picked up request " << req.request_id;
			int max_gen = req.max_tokens > 0 ? req.max_tokens : config.default_max_tokens;
			float temp = req.temperature > 0.0f ? req.temperature : config.default_temperature;
			int top_k = req.top_k > 0 ? req.top_k : config.default_top_k;
			float top_p = req.top_p > 0.0f && req.top_p < 1.0f ? req.top_p : config.default_top_p;
			scheduler->add_request(req.request_id, req.prompt, req.token_ids, max_gen, temp, top_k, top_p);
		}

		std::vector<StepResult> results = scheduler->step();

		for (const auto &res : results) {
			if (coordinator->is_cancelled(res.request_id))
				continue;

			auto client_queue = coordinator->get_queue(res.request_id);
			if (client_queue) {
				client_queue->push(res);
			}
		}
	}

	LOG_INFO("Engine") << "Shutdown.";
}

int main(int argc, char **argv) {
	Config config = Config::parse_args(argc, argv);
	try {
		std::string server_address = config.server_address;

		LOG_INFO("Main") << "--- Initializing tinylm ---";
		LOG_INFO("Main") << "Prefill model: " << config.prefill_model_path;
		LOG_INFO("Main") << "Decode model: " << config.decode_model_path;
		LOG_INFO("Main") << "Tokenizer: " << config.tokenizer_path;
		torch::Device device(torch::kCPU);

		Tokenizer tokenizer(config.tokenizer_path);
		Loader prefill_loader(config.prefill_model_path, device);
		Loader decode_loader(config.decode_model_path, device);

		LOG_INFO("Main") << "Max batch size: " << config.max_batch_size;

		auto scheduler = std::make_shared<Scheduler>(prefill_loader, decode_loader, tokenizer, config.max_batch_size);
		auto coordinator = std::make_shared<Coordinator>();

		std::thread engine_thread(RunEngineLoop, scheduler, coordinator, config);

		InferenceServiceImpl service(coordinator, tokenizer);

		grpc::ServerBuilder builder;
		builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
		builder.RegisterService(&service);

		std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
		LOG_INFO("Main") << "Server listening on " << server_address;

		server->Wait();

		coordinator->stop_requested = true;
		coordinator->request_queue.shutdown();
		engine_thread.join();
		LOG_INFO("Main") << "Engine thread joined. Exiting.";

	} catch (const std::exception &e) {
		LOG_ERROR("Main") << "Fatal error: " << e.what();
		return 1;
	}

	return 0;
}
