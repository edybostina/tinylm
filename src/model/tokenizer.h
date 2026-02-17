#pragma once
#include <string>
#include <vector>

namespace sentencepiece {
class SentencePieceProcessor;
}

class Tokenizer {
   private:
	std::unique_ptr<sentencepiece::SentencePieceProcessor> processor_;

   public:
	Tokenizer(const std::string &model_path);
	~Tokenizer();
	std::vector<int> encode(const std::string &text, bool add_bos = true) const;
	std::string decode(const std::vector<int> &ids) const;
	std::string decode(int id) const;

	int get_vocab_size() const;
	int get_bos_id() const;
	int get_eos_id() const;
	int get_pad_id() const;
};
