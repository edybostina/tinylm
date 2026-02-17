#include "tokenizer.h"
#include <sentencepiece_processor.h>
#include <stdexcept>

Tokenizer::Tokenizer(const std::string &model_path) {
	processor_ = std::make_unique<sentencepiece::SentencePieceProcessor>();
	const auto status = processor_->Load(model_path);
	if (!status.ok()) {
		throw std::runtime_error(status.ToString());
	}
}

Tokenizer::~Tokenizer() = default;

int Tokenizer::get_vocab_size() const {
	return processor_->GetPieceSize();
}

int Tokenizer::get_bos_id() const {
	return processor_->bos_id();
}

int Tokenizer::get_eos_id() const {
	return processor_->eos_id();
}

int Tokenizer::get_pad_id() const {
	return processor_->pad_id();
}

std::vector<int> Tokenizer::encode(const std::string &text, bool add_bos) const {
	std::vector<int> ids;
	processor_->Encode(text, &ids);
	if (add_bos) {
		ids.insert(ids.begin(), get_bos_id());
	}
	return ids;
}

std::string Tokenizer::decode(const std::vector<int> &ids) const {
	std::string result;
	return processor_->Decode(ids, &result).ok() ? result : "";
}
std::string Tokenizer::decode(int id) const {
	std::string piece = processor_->IdToPiece(id);

	const std::string sp_space = "\xe2\x96\x81";

	size_t pos = 0;
	while ((pos = piece.find(sp_space, pos)) != std::string::npos) {
		piece.replace(pos, sp_space.length(), " ");
		pos += 1;
	}

	return piece;
}
