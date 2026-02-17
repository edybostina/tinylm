import torch
import torch.nn as nn


class TinyModel(nn.Module):
    def __init__(self, vocab_size=250000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 16)
        self.linear = nn.Linear(16, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        return self.linear(x)


model = TinyModel().eval()
scripted = torch.jit.script(model)
scripted.save("tiny.pt")
