#pragma once

#include <string>
#include <vector>
#include <torch/script.h>

class Loader {
   private:
	torch::jit::script::Module model_;
	torch::Device device_;

   public:
	Loader(const std::string &model_path, torch::Device device);
	torch::Device get_device() const;
	torch::IValue forward(const std::vector<torch::IValue> &inputs);
};
