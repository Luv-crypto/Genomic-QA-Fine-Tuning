#!/usr/bin/env python
# ──────────────────────────────────────────────────────────────────────────────
# eval.py ― quick-and-clean loss evaluator for any PEFT (LoRA/QLoRA) checkpoint
# Tested on:
#   torch==2.2.1+cu121         transformers==4.52.4
#   peft==0.11.1               accelerate==1.7.0
#   python 3.10 / CUDA 12.1
# ──────────────────────────────────────────────────────────────────────────────

import argparse, json, os, pathlib, time
from typing import List, Dict

import torch
from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

# ------------------------------------------------------------------- PEFT shim
from peft import PeftModel                 # always exists
try:                                       # PEFT ≤ 0.5.x exported this helper
    from peft import is_peft_model         # noqa: F401
except ImportError:                        # PEFT ≥ 0.6.x – recreate it
    def is_peft_model(m):                  # type: ignore  # minimal wrapper
        return isinstance(m, PeftModel)


# ──────────────────────────────── helpers ────────────────────────────────────
def load_dataset(path: str) -> List[Dict[str, str]]:
    """`eval.json` → List[{"prompt": .., "answer": ..}]"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list) and all(
        "question" in d and "answer" in d for d in data
    ), "Dataset must be a list of objects with 'prompt' and 'answer'"
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

        labels = input_ids.clone()
        labels[labels == pad_id] = -100          # ignore padding tokens
        for i, b in enumerate(batch):            # ignore prompt loss
            p_len = tokenizer(b["question"]).input_ids.__len__()
            labels[i, :p_len] = -100

        return {
            "input_ids":      input_ids,
            "attention_mask": attn_mask,
            "labels":         labels,
        }

    return collate


# ─────────────────────────────── main routine ────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Simple eval for PEFT checkpoints")
    ap.add_argument("--model",   required=True, help="folder or HF repo id")
    ap.add_argument("--dataset", required=True, help="eval.json path")
    ap.add_argument("--batch",   type=int, default=4)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--offload", action="store_true",
                    help="CPU/disk off-loading when VRAM is low")
    args = ap.parse_args()

    device_map = "auto" if args.offload else {"": 0}
    offload_kw = dict(
        offload_folder="offload",
        offload_state_dict=True,
    ) if args.offload else {}

    print(f"Loading model from {args.model} …")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        low_cpu_mem_usage=True,
        **offload_kw,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token      # make sure padding exists

    data = load_dataset(args.dataset)
    dl = DataLoader(
        data,
        batch_size=args.batch,
        shuffle=False,
        collate_fn=build_collate(tokenizer, args.max_len),
    )

    model.eval()
    losses, t0 = [], time.time()
    with torch.no_grad():
        for step, batch in enumerate(dl, 1):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss.detach().float()
            losses.append(loss.item())

            if step % 20 == 0 or step == len(dl):
                print(f"[{step:>4}/{len(dl)}]  loss={loss.item():.4f}")

    avg_loss = sum(losses) / len(losses)
    ppl       = torch.exp(torch.tensor(avg_loss)).item()
    dur       = time.time() - t0

    metrics = {
        "loss":        avg_loss,
        "perplexity":  ppl,
        "examples":    len(data),
        "batch_size":  args.batch,
        "max_len":     args.max_len,
        "seconds":     dur,
        "model":       pathlib.Path(args.model).name,
        "timestamp":   time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_path = "eval_metrics.json"
    json.dump(metrics, open(out_path, "w"), indent=2)
    print("\n✔ Finished\n", json.dumps(metrics, indent=2))
    print(f"Metrics written to {out_path}")


if __name__ == "__main__":
    main()
