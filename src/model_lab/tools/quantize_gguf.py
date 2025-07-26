#!/usr/bin/env python
"""
quantize_gguf.py – wrap llama.cpp quantize, mode from params.yaml
"""
import argparse, subprocess, pathlib, yaml, sys

# ─── load config ─────────────────────────────────────────────
cfg = yaml.safe_load(open("params.yaml"))
ci  = cfg.get("ci", {})
env = ci.get("env", {})

LLAMA_CPP_DIR = pathlib.Path(env["LLAMA_CPP_DIR"]).resolve()
quant_bin     = LLAMA_CPP_DIR / "build" / "bin" / "Release" / "llama-quantize.exe"
if not quant_bin.exists():
    sys.exit(f"❌ quantize binary not found at {quant_bin}")

mode = ci.get("quant_mode", "q4_k_m")  # YAML-driven

# ─── CLI args ─────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("in_file",  help="input .gguf")
p.add_argument("out_file",  help="output .gguf")
p.add_argument("quant_type", help= "Quantization Mode")
_ = p.parse_args()  # we ignore --mode

IN  = p.parse_args().in_file
OUT = p.parse_args().out_file

# ─── run quantizer ────────────────────────────────────────────
pathlib.Path(OUT).parent.mkdir(parents=True, exist_ok=True)
cmd = [str(quant_bin), IN, OUT, mode]
print("Running:", " ".join(cmd))
subprocess.check_call(cmd)
print("✓ quantized:", OUT, "mode=", mode)
