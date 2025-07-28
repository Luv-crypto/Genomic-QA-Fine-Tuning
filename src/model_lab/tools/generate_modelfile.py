#!/usr/bin/env python
"""Generate an Ollama Modelfile pointing at a GGUF model."""
import argparse
from pathlib import Path

TPL = """FROM {base}
ADAPTER /model/model.gguf
"""


def main():
    ap = argparse.ArgumentParser(description="Create Modelfile for Ollama")
    ap.add_argument("gguf", help="Path to gguf file (unused; for pipeline tracking)")
    ap.add_argument("--base", default="llama2", help="Base model family")
    ap.add_argument("--out", default="Modelfile", help="Output Modelfile path")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    template = Path('Modelfile.template')
    text = template.read_text() if template.exists() else TPL
    out.write_text(text.format(base=args.base))
    print(f"\u2713 Modelfile written: {out}")

if __name__ == "__main__":
    main()
