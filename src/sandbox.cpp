#include <iostream>
#include <vector>
#include <string>
#include "model/tokenizer.h"
#include "model/loader.h"

int main() {
	try {
		torch::Device device(torch::kCPU);

		Tokenizer tokenizer("sentencepiece_models/sentencepiece.bpe.model");
		Loader loader("pytorch_weights/tiny.pt", device);
		std::cout << "My Tokenizer Vocab Size is: " << tokenizer.get_vocab_size() << "\n";
		std::string prompt = "Hello, tinylm!";
		std::vector<int> ids = tokenizer.encode(prompt, true);	// true = add BOS
		std::cout << "Tokenized prompt into " << ids.size() << " tokens.\n";
		std::vector<int64_t> ids_64(ids.begin(), ids.end());

		auto options = torch::TensorOptions().dtype(torch::kInt64);

		torch::Tensor input_tensor =
			torch::from_blob(ids_64.data(), {1, static_cast<long long>(ids_64.size())}, options).to(device);
		std::vector<torch::IValue> inputs = {input_tensor};
		torch::IValue output = loader.forward(inputs);

		torch::Tensor out_tensor = output.toTensor();
		std::cout << "Success! Output tensor shape: " << out_tensor.sizes() << "\n";

	} catch (const std::exception& e) {
		std::cerr << "Fatal Error: " << e.what() << "\n";
	}

	return 0;
}
