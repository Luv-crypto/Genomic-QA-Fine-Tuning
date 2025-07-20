#!/usr/bin/env python
"""
merge_adapters.py  – merge LoRA adapter into base weights
Called by DVC; paths come from CLI (dvc.yaml).
"""

import argparse, shutil, pathlib, torch, json
from transformers import AutoModelForCausalLM
from peft import PeftModel

p = argparse.ArgumentParser()
p.add_argument("--base",    required=True)   # folder with base weights
p.add_argument("--adapter", required=True)   # folder with adapter weights
p.add_argument("--out",     required=True)   # output merged folder
args = p.parse_args()

base_path    = pathlib.Path(args.base)
adapter_path = pathlib.Path(args.adapter)
out_path     = pathlib.Path(args.out)

print(f"Loading base  : {base_path}")
model = AutoModelForCausalLM.from_pretrained(
    base_path, torch_dtype=torch.float16, device_map="cpu"
)

print(f"Inject adapter: {adapter_path}")
model = PeftModel.from_pretrained(model, adapter_path)

print("Merging & unloading …")
merged = model.merge_and_unload()

print(f"Saving merged weights → {out_path}")
merged.save_pretrained(out_path, safe_serialization=True)
shutil.copyfile(base_path / "config.json", out_path / "config.json")
print("✓ merge complete")
