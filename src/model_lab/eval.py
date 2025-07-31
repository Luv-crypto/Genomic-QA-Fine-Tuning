#!/usr/bin/env python
# ──────────────────────────────────────────────────────────────────────────────
# eval.py ― quick-and-clean loss evaluator for any PEFT (LoRA/QLoRA) checkpoint
# This version forces purely local loading from a folder path.
# ──────────────────────────────────────────────────────────────────────────────

import argparse, json, pathlib, time
from typing import List, Dict
from datetime import datetime
import torch
from torch.utils.data import DataLoader

# ─── Monkey-patch HF Hub validation so it never rejects Windows paths ─────────
import huggingface_hub.utils._validators as _validators
_validators.validate_repo_id = lambda *args, **kwargs: None

from transformers import AutoTokenizer, AutoModelForCausalLM

# ------------------------------------------------------------------- PEFT shim
from peft import PeftModel
try:
    # PEFT ≤0.5.x
    from peft import is_peft_model
except ImportError:
    def is_peft_model(m): return isinstance(m, PeftModel)


def load_dataset(path: str) -> List[Dict[str, str]]:
    """`eval.json` → List[{"question":..., "answer":...}]"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list) and all(
        "question" in d and "answer" in d for d in data
    ), "Dataset must be a list of {'question','answer'} objects"
    return data


def build_collate(tokenizer, max_len: int):
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def collate(batch):
        texts = [b["question"] + b["answer"] for b in batch]
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )
        input_ids = enc["input_ids"]
        attn_mask = enc["attention_mask"]

        # prepare labels (mask prompt)
        labels = input_ids.clone()
        labels[labels == pad_id] = -100
        for i, b in enumerate(batch):
            prompt_len = len(tokenizer(b["question"]).input_ids)
            labels[i, :prompt_len] = -100

        return {"input_ids": input_ids,
                "attention_mask": attn_mask,
                "labels": labels}

    return collate


def main():
    ap = argparse.ArgumentParser(description="Local eval for PEFT checkpoints")
    ap.add_argument("--model",   required=True, help="path to your run folder")
    ap.add_argument("--dataset", required=True, help="path to eval.json")
    ap.add_argument("--batch",   type=int, default=4)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--offload", action="store_true",
                    help="offload state_dict to disk if VRAM is low")
    args = ap.parse_args()

    # 1) Resolve absolute path so HF sees a real directory
    model_dir = pathlib.Path(args.model).expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Eval model folder not found: {model_dir}")
    print(f"Loading model from {model_dir!r} …")

    # 2) Configure offloading if requested
    device_map = "auto" if args.offload else {"": 0}
    offload_kwargs = {}
    if args.offload:
        offload_kwargs = dict(
            offload_folder=str(model_dir / "offload"),
            offload_state_dict=True,
        )

    # 3) Load model & tokenizer purely from disk
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        low_cpu_mem_usage=True,
        **offload_kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4) DataLoader setup
    data = load_dataset(args.dataset)
    dl = DataLoader(
        data,
        batch_size=args.batch,
        shuffle=False,
        collate_fn=build_collate(tokenizer, args.max_len),
    )

    # 5) Run evaluation
    model.eval()
    losses, start = [], time.time()
    with torch.no_grad():
        for i, batch in enumerate(dl, 1):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch)
            losses.append(out.loss.item())
            if i % 20 == 0 or i == len(dl):
                print(f"[{i:>4}/{len(dl)}] loss={losses[-1]:.4f}")

    # 6) Report metrics
    avg_loss = sum(losses) / len(losses)
    ppl      = torch.exp(torch.tensor(avg_loss)).item()
    duration = time.time() - start

    metrics = {
        "loss":       avg_loss,
        "perplexity": ppl,
        "examples":   len(data),
        "batch_size": args.batch,
        "max_len":    args.max_len,
        "seconds":    duration,
        "model":      model_dir.name,
        "timestamp":  time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # ---------- 1) canonical (stable) file  ---------------------------
    family   = model_dir.parts[-2]              # …/candidates/<family>/<run>_merged
    run_name = model_dir.name.replace("_merged", "")
    metrics_dir   = pathlib.Path("metrics") / family
    archive_dir   = metrics_dir / "archive"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)

    stable_path   = metrics_dir / f"{run_name}.json"
    archive_path  = archive_dir / f"{run_name}_{datetime.now():%Y%m%dT%H%M%S}.json"

    with open(stable_path, "w")  as f: json.dump(metrics, f, indent=2)
    with open(archive_path, "w") as f: json.dump(metrics, f, indent=2)

    print(f"✓ Metrics written: {stable_path}  (stable)")
    print(f"✓ Archived copy : {archive_path}")

if __name__ == "__main__":
    main()
