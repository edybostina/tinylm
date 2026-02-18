#pragma once

#include <string>
#include <iostream>
#include <vector>

struct Config {
	std::string model_path = "pytorch_weights/tiny.pt";
	std::string tokenizer_path = "sentencepiece_models/sentencepiece.bpe.model";
	std::string server_address = "0.0.0.0:50051";
	int default_max_tokens = 50;

	float default_temperature = 1.0f;
	int default_top_k = 0;
	float default_top_p = 1.0f;

	int kv_num_blocks = 1000;
	int kv_num_heads = 2;
	int kv_head_dim = 64;
	int kv_block_size = 16;

	int max_batch_size = 32;

	static Config parse_args(int argc, char **argv) {
		Config config;
		std::vector<std::string> args(argv + 1, argv + argc);

		for (size_t i = 0; i < args.size(); ++i) {
			if (args[i] == "--model" && i + 1 < args.size()) {
				config.model_path = args[++i];
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
			} else if (args[i] == "--kv_num_blocks" && i + 1 < args.size()) {
				config.kv_num_blocks = std::stoi(args[++i]);
			} else if (args[i] == "--kv_num_heads" && i + 1 < args.size()) {
				config.kv_num_heads = std::stoi(args[++i]);
			} else if (args[i] == "--kv_head_dim" && i + 1 < args.size()) {
				config.kv_head_dim = std::stoi(args[++i]);
			} else if (args[i] == "--kv_block_size" && i + 1 < args.size()) {
				config.kv_block_size = std::stoi(args[++i]);
			} else if (args[i] == "--max_batch_size" && i + 1 < args.size()) {
				config.max_batch_size = std::stoi(args[++i]);
			} else if (args[i] == "--help") {
				std::cout << "Usage: ./sandbox [--model <path>] [--tokenizer <path>] [--port <port>]\n"
							 "                [--max_tokens <num>] [--temperature <float>] [--top_k <int>] [--top_p "
							 "<float>]\n"
							 "                [--kv_num_blocks <int>] [--kv_num_heads <int>] [--kv_head_dim <int>]\n"
							 "                [--kv_block_size <int>] [--max_batch_size <int>]\n";
				exit(0);
			}
		}
		return config;
	}
};
