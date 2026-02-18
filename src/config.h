#pragma once

#include <string>
#include <iostream>
#include <vector>

struct Config {
	std::string prefill_model_path = "pytorch_weights/prefill.pt";
	std::string decode_model_path = "pytorch_weights/decode.pt";
	std::string tokenizer_path = "pytorch_weights/tokenizer.model";
	std::string server_address = "0.0.0.0:50051";
	int default_max_tokens = 50;

	float default_temperature = 1.0f;
	int default_top_k = 0;
	float default_top_p = 1.0f;

	int max_batch_size = 32;

	static Config parse_args(int argc, char **argv) {
		Config config;
		std::vector<std::string> args(argv + 1, argv + argc);

		for (size_t i = 0; i < args.size(); ++i) {
			if (args[i] == "--prefill_model" && i + 1 < args.size()) {
				config.prefill_model_path = args[++i];
			} else if (args[i] == "--decode_model" && i + 1 < args.size()) {
				config.decode_model_path = args[++i];
			} else if (args[i] == "--tokenizer" && i + 1 < args.size()) {
				config.tokenizer_path = args[++i];
			} else if (args[i] == "--port" && i + 1 < args.size()) {
				config.server_address = "0.0.0.0:" + args[++i];
			} else if (args[i] == "--max_tokens" && i + 1 < args.size()) {
				config.default_max_tokens = std::stoi(args[++i]);
			} else if (args[i] == "--temperature" && i + 1 < args.size()) {
				config.default_temperature = std::stof(args[++i]);
			} else if (args[i] == "--top_k" && i + 1 < args.size()) {
				config.default_top_k = std::stoi(args[++i]);
			} else if (args[i] == "--top_p" && i + 1 < args.size()) {
				config.default_top_p = std::stof(args[++i]);
			} else if (args[i] == "--max_batch_size" && i + 1 < args.size()) {
				config.max_batch_size = std::stoi(args[++i]);
			} else if (args[i] == "--help") {
				std::cout << "Usage: ./tinylm [--prefill_model <path>] [--decode_model <path>]\n"
							 "                [--tokenizer <path>] [--port <port>]\n"
							 "                [--max_tokens <num>] [--temperature <float>]\n"
							 "                [--top_k <int>] [--top_p <float>]\n"
							 "                [--max_batch_size <int>]\n";
				exit(0);
			}
		}
		return config;
	}
};
