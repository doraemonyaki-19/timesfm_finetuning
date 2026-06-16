"""Convert safetensors to torch .pt format (smaller peak memory for loading)."""
import sys
from safetensors import safe_open
import importlib

# Import torch carefully
torch = importlib.import_module("torch")

src = sys.argv[1]  # e.g. checkpoints/run_glia_1/best/model.safetensors
dst = sys.argv[2]  # e.g. checkpoints/run_glia_1/best/model.pt

print(f"Converting {src} -> {dst}")
state_dict = {}
with safe_open(src, framework="pt", device="cpu") as f:
  for key in f.keys():
    state_dict[key] = f.get_tensor(key)

torch.save(state_dict, dst)
print(f"Saved {dst}")
