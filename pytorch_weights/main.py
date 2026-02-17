import torch
import torch.nn as nn


class TinyModel(nn.Module):
    # Note: Use the vocab_size from your SentencePiece model!
    # 32000 is standard for Llama, but yours might be different.
    def __init__(self, vocab_size=32000):
        super().__init__()
        # 1. The Lookup Table: Turns an integer ID into a vector of size 16
        self.embedding = nn.Embedding(vocab_size, 16)
        # 2. The Linear Layer: Processes vectors of size 16
        self.linear = nn.Linear(16, 16)

    def forward(self, x):
        # Input 'x' is [1, 7] (Integers)
        x = self.embedding(x)
        # Now 'x' is [1, 7, 16] (Floats)
        return self.linear(x)


ACTUAL_VOCAB_SIZE = 250000

model = TinyModel(vocab_size=ACTUAL_VOCAB_SIZE).eval()
scripted = torch.jit.script(model)
scripted.save("tiny.pt") layer!")
