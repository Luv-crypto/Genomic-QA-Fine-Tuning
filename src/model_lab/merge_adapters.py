#!/usr/bin/env python
"""
merge_adapters.py  – merge LoRA adapter into base weights
Called by DVC; paths come from CLI (dvc.yaml).
"""

import argparse, pathlib, torch
from transformers import AutoModelForCausalLM, AutoTokenizer # <- Import AutoTokenizer
from peft import PeftModel

p = argparse.ArgumentParser()
p.add_argument("--base",    required=True)  # folder with base weights
p.add_argument("--adapter", required=True)  # folder with adapter weights
p.add_argument("--out",     required=True)  # output merged folder
args = p.parse_args()

base_path    = pathlib.Path(args.base)
adapter_path = pathlib.Path(args.adapter)
out_path     = pathlib.Path(args.out)

print(f"Loading base   : {base_path}")
model = AutoModelForCausalLM.from_pretrained(
    base_path, torch_dtype=torch.float16, device_map="cpu"
)

# --- ADD THIS ---
print(f"Loading tokenizer: {base_path}")
tokenizer = AutoTokenizer.from_pretrained(base_path)
# ----------------

print(f"Inject adapter: {adapter_path}")
model = PeftModel.from_pretrained(model, adapter_path)

print("Merging & unloading …")
merged = model.merge_and_unload()

print(f"Saving merged model & tokenizer → {out_path}")
merged.save_pretrained(out_path, safe_serialization=True)
tokenizer.save_pretrained(out_path) # <- Save the tokenizer
# -----------------------------------

# You can now remove the manual copy of config.json, as save_pretrained handles it.
# shutil.copyfile(base_path / "config.json", out_path / "config.json")

print("✓ merge complete")