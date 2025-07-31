# #!/usr/bin/env python
# """
# Generate a customized Ollama Modelfile from a template using
# command-line arguments.
# """
# import argparse
# from pathlib import Path

# def main():
#     """Main function to generate the Modelfile."""
#     parser = argparse.ArgumentParser(
#         description="Generate an Ollama Modelfile from a template."
#     )
#     # DVC dependency tracking argument (value is not used in the script)
#     parser.add_argument(
#         "gguf_path", help="Path to the GGUF file for DVC dependency tracking."
#     )
#     # Arguments for populating the template
#     parser.add_argument("--base", required=True, help="Base model name for 'FROM' instruction (e.g., qwen:7b).")
#     parser.add_argument("--template", default="Modelfile.template", help="Path to the Modelfile template.")
#     parser.add_argument("--ctx_length", type=int, default=4096, help="Context length for the model.")
#     parser.add_argument("--system_prompt", default="", help="The system prompt to embed in the Modelfile.")
#     parser.add_argument("--out", required=True, help="Output Modelfile path.")
#     args = parser.parse_args()

#     # --- Read Template ---
#     template_path = Path(args.template)
#     if not template_path.exists():
#         raise FileNotFoundError(f"Template file not found at: {template_path}")
#     template_text = template_path.read_text()

#     # --- Prepare Substitutions ---
#     # These keys must match the placeholders in your Modelfile.template
#     substitutions = {
#         "base": args.base,
#         "ctx_length": args.ctx_length,
#         "system_prompt": args.system_prompt,
#     }

#     # --- Generate Modelfile Content ---
#     try:
#         content = template_text.format(**substitutions)
#     except KeyError as e:
#         print(
#             f"Error: The template '{template_path}' requires a key that was not provided as an argument.\n"
#             f"Missing key: {e}\n"
#             f"Available substitution keys: {list(substitutions.keys())}"
#         )
#         exit(1)

#     # --- Write Output ---
#     out_path = Path(args.out)
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     out_path.write_text(content)

#     print(f"\u2713 Modelfile written to: {out_path}")

# if __name__ == "__main__":
#     main()


import argparse
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description="Generate an Ollama Modelfile for GGUF-only models.")
    p.add_argument("gguf_path", help="Path to the GGUF file for DVC tracking.")
    p.add_argument("--template", default="Modelfile.template", help="Template path.")
    p.add_argument("--ctx_length", type=int, required=True, help="Context window length.")
    p.add_argument("--system_prompt", required=True, help="System prompt text.")
    p.add_argument("--out", required=True, help="Output Modelfile path.")
    args = p.parse_args()

    # Read template
    tpl = Path(args.template)
    if not tpl.exists():
        raise FileNotFoundError(f"Template not found: {tpl}")
    template_text = tpl.read_text()

    # Extract GGUF filename
    gguf_filename = Path(args.gguf_path).name

    # Perform substitutions
    subs = {
        "gguf_filename": gguf_filename,
        "ctx_length":    args.ctx_length,
        "system_prompt": args.system_prompt,
    }
    content = template_text.format(**subs)

    # Write output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content)
    print(f"✓ Modelfile written to: {out_path}")

if __name__ == "__main__":
    main()
