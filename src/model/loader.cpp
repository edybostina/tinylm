#include "loader.h"
#include <stdexcept>

Loader::Loader(const std::string &model_path, torch::Device device) : device_(device) {
	try {
		model_ = torch::jit::load(model_path, device_);
		model_.eval();
	} catch (const c10::Error &e) {
		throw std::runtime_error("Failed to load model: " + std::string(e.what()));
	}
}

torch::Device Loader::get_device() const {
	return device_;
}

torch::IValue Loader::forward(const std::vector<torch::IValue> &inputs) {
	torch::InferenceMode guard;

	return model_.forward(inputs);
}
