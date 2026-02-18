import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
import os
import shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print(f"Loading {model_id}...")

tokenizer = AutoTokenizer.from_pretrained(model_id)
vocab_size = tokenizer.vocab_size
print(f"Vocabulary Size: {vocab_size}")

model = AutoModelForCausalLM.from_pretrained(
    model_id, use_cache=False, torch_dtype=torch.float32
)
model.eval()

dest_model = os.path.join(SCRIPT_DIR, "tokenizer.model")
print("Downloading tokenizer.model from hub...")
cached_path = hf_hub_download(repo_id=model_id, filename="tokenizer.model")
shutil.copy(cached_path, dest_model)
print(f"Tokenizer saved to {dest_model}")


class Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        attention_mask = torch.ones_like(input_ids)
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask)
        return outputs.logits


print("Tracing the model (this takes a minute)...")
wrapped_model = Wrapper(model)

dummy_input = torch.randint(0, vocab_size, (1, 10))

with torch.no_grad():
    traced_model = torch.jit.trace(wrapped_model, dummy_input)

pt_path = os.path.join(SCRIPT_DIR, "tiny.pt")
print(f"Saving {pt_path}...")
traced_model.save(pt_path)
