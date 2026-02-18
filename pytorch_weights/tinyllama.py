import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
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
    model_id, dtype=torch.float32
)
model.config.use_cache = True
model.eval()

NUM_LAYERS = model.config.num_hidden_layers       # 22
NUM_KV_HEADS = model.config.num_key_value_heads    # 4
HEAD_DIM = model.config.hidden_size // model.config.num_attention_heads  # 64
print(f"num_layers={NUM_LAYERS}, num_kv_heads={
      NUM_KV_HEADS}, head_dim={HEAD_DIM}")

dest_model = os.path.join(SCRIPT_DIR, "tokenizer.model")
print("Downloading tokenizer.model from hub...")
cached_path = hf_hub_download(repo_id=model_id, filename="tokenizer.model")
shutil.copy(cached_path, dest_model)
print(f"Tokenizer saved to {dest_model}")


def extract_kv(past_key_values):
    """Return stacked (keys, values) tensors from either DynamicCache or tuple."""
    if hasattr(past_key_values, "key_cache"):
        # transformers >= 4.36 DynamicCache
        return torch.stack(list(past_key_values.key_cache)), \
            torch.stack(list(past_key_values.value_cache))
    return torch.stack([kv[0] for kv in past_key_values]), \
        torch.stack([kv[1] for kv in past_key_values])


class PrefillWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        attention_mask = torch.ones_like(input_ids)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        keys, values = extract_kv(outputs.past_key_values)
        return outputs.logits, keys, values


class DecodeWrapper(nn.Module):
    def __init__(self, model, num_layers):
        super().__init__()
        self.model = model
        self.num_layers = num_layers

    def forward(self, input_ids, key_cache, value_cache):
        past_kv = DynamicCache()
        for i in range(self.num_layers):
            past_kv.update(key_cache[i], value_cache[i], i)

        past_len = key_cache.size(3)
        attention_mask = torch.ones(
            1, past_len + input_ids.size(1),
            dtype=torch.long, device=input_ids.device,
        )
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_kv,
            use_cache=True,
        )
        new_keys, new_values = extract_kv(outputs.past_key_values)
        return outputs.logits, new_keys, new_values


print("Tracing prefill model...")
prefill_wrapper = PrefillWrapper(model)
dummy_prefill = torch.randint(0, vocab_size, (1, 10))

with torch.no_grad():
    prefill_traced = torch.jit.trace(prefill_wrapper, dummy_prefill)

prefill_path = os.path.join(SCRIPT_DIR, "prefill.pt")
prefill_traced.save(prefill_path)
print(f"Saved {prefill_path}")

print("Tracing decode model...")
decode_wrapper = DecodeWrapper(model, NUM_LAYERS)
dummy_decode_ids = torch.randint(0, vocab_size, (1, 1))
dummy_key_cache = torch.randn(NUM_LAYERS, 1, NUM_KV_HEADS, 10, HEAD_DIM)
dummy_val_cache = torch.randn(NUM_LAYERS, 1, NUM_KV_HEADS, 10, HEAD_DIM)

with torch.no_grad():
    decode_traced = torch.jit.trace(
        decode_wrapper, (dummy_decode_ids, dummy_key_cache, dummy_val_cache)
    )

decode_path = os.path.join(SCRIPT_DIR, "decode.pt")
decode_traced.save(decode_path)
print(f"Saved {decode_path}")

print("Done! You now have 'tokenizer.model', 'prefill.pt', and 'decode.pt'.")
