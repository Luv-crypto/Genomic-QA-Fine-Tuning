#!/usr/bin/env python
"""
convert_to_gguf.py – HF→GGUF conversion via llama.cpp,
reads llama.cpp venv info from params.yaml
"""

import argparse, subprocess, pathlib, os, sys, yaml

# ───────────────── load env from params.yaml ─────────────────
cfg = yaml.safe_load(open("params.yaml"))
env = cfg.get("ci", {}).get("env", {})

LLAMA_CPP_DIR = pathlib.Path(env.get("LLAMA_CPP_DIR", "llama.cpp")).resolve()
PYTHON        = pathlib.Path(env.get("LLAMA_PY", sys.executable)).resolve()

# ───────────────── locate converter script ───────────────────
candidates = [
    LLAMA_CPP_DIR / "convert-hf-to-gguf.py",
    LLAMA_CPP_DIR / "convert" / "convert-hf-to-gguf.py",
]
CONVERT_SCRIPT = next((p for p in candidates if p.exists()), None)
if CONVERT_SCRIPT is None:
    sys.exit(f"❌ convert-hf-to-gguf.py not found under {LLAMA_CPP_DIR}")

# ────────────────── CLI args ─────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--in_model",  required=True,
               help="path to the merged HF model folder")
p.add_argument("--out_file",  required=True,
               help="path to write the .gguf file")
args = p.parse_args()

# ensure output directory exists
pathlib.Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)

# ────────────────── run conversion ────────────────────────────
cmd = [
    str(PYTHON), str(CONVERT_SCRIPT),
    "--model",   args.in_model,
    "--outfile", args.out_file,
]
print("Running:", " ".join(cmd))
subprocess.check_call(cmd)
print("✓ gguf created:", args.out_file)
