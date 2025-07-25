#!/usr/bin/env python
"""
model_lab.train   (YAML-only configuration)

• reads every knob from params.yaml
• saves weights to  {output_root}/{timestamp}-{run_name}/
• writes metrics/eval_local.json
• defers tensor creation to batch time (low RAM)
• offloads state dict to disk if needed (low GPU VRAM)
• releases unused memory after training
"""

from __future__ import annotations
import json, time, pathlib, datetime as dt, yaml, gc

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling,
    EarlyStoppingCallback, BitsAndBytesConfig, TrainerCallback,
)
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

# ─────────────────────────────────────────────────────────────
# 0. load YAML + cast numeric types
# ─────────────────────────────────────────────────────────────
CFG = yaml.safe_load(open("params.yaml"))
# cast strings → correct types
CFG["training"]["lr"]           = float(CFG["training"]["lr"])
CFG["training"]["batch_size"]   = int(CFG["training"]["batch_size"])
CFG["training"]["grad_accum"]   = int(CFG["training"]["grad_accum"])
CFG["training"]["epochs"]       = int(CFG["training"]["epochs"])

RUN   = CFG["run_name"]
STAMP = dt.datetime.now().strftime("%Y-%m-%dT%H-%M")
ROOT  = pathlib.Path(CFG["output_root"])
OUT   = (ROOT / f"{STAMP}-{RUN}").resolve()
OFFLD = OUT / "offload"
# create both directories
OUT.mkdir(parents=True, exist_ok=True)
OFFLD.mkdir(parents=True, exist_ok=True)

# shorthands
BASE  = CFG["base_model"]
DS    = CFG["dataset"]
TOK   = CFG["tokenizer"]
Q     = CFG["quantization"]
TRAIN = CFG["training"]
LORA  = CFG["lora"]


# ─────────────────────────────────────────────────────────────
# 1. model + tokenizer
# ─────────────────────────────────────────────────────────────
bnb_cfg = None
if Q["use_4bit"]:
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=Q["quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, Q["compute_dtype"]),
        bnb_4bit_use_double_quant=Q["double_quant"],
    )

print(f"[{RUN}] loading {BASE['repo']} (rev {BASE['revision']})")
model = AutoModelForCausalLM.from_pretrained(
    BASE["repo"],
    revision=BASE["revision"],
    trust_remote_code=True,
    device_map="auto",
    low_cpu_mem_usage=True,
    offload_folder=str(OFFLD),
    offload_state_dict=True,
    quantization_config=bnb_cfg,
)
if Q["use_4bit"]:
    model = prepare_model_for_kbit_training(model)

tokenizer = AutoTokenizer.from_pretrained(
    BASE["repo"],
    revision=BASE["revision"],
    trust_remote_code=True,
    use_fast=False,
)
tokenizer.pad_token    = tokenizer.pad_token or tokenizer.eos_token
tokenizer.padding_side = "right"

# ─────────────────────────────────────────────────────────────
# 2. LoRA adapter injection
# ─────────────────────────────────────────────────────────────
peft_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=LORA["r"],
    lora_alpha=LORA["alpha"],
    lora_dropout=LORA["dropout"],
    target_modules=LORA["target_modules"],
    bias="lora_only",
    modules_to_save=["lm_head"],
    use_rslora=True,
)
model = get_peft_model(model, peft_cfg)
model.print_trainable_parameters()

# ─────────────────────────────────────────────────────────────
# 3. dataset mapping (defer tensor conversion)
# ─────────────────────────────────────────────────────────────
def fmt(ex):
    return {"text": [f"### Human: {q}\n### Assistant: {a}"
                     for q, a in zip(ex["question"], ex["answer"])]}

print(f"[{RUN}] loading dataset {DS['path']}")
raw = load_dataset("json", data_files=DS["path"])
raw = raw.map(fmt, batched=True, remove_columns=["question", "answer"])

train_raw = raw["train"].select(range(DS["sample_size"]))
eval_raw  = raw["train"].select(range(DS["sample_size"],
                                      DS["sample_size"] + DS["eval_size"]))

def tok(batch):
    enc = tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=TOK["max_length"],
    )
    labels = []
    for ids in enc["input_ids"]:
        lab = list(ids)
        try:
            pos = lab.index(tokenizer.encode("Assistant:")[0]) + 9
            lab[:pos] = [-100] * pos
        except ValueError:
            lab = [-100] * len(lab)
        labels.append(lab)
    return {
        "input_ids":     enc["input_ids"],       # list[int]
        "attention_mask": enc["attention_mask"],  # list[int]
        "labels":         labels                 # list[int]
    }

train_ds = train_raw.map(tok,
                         batched=True,
                         batch_size=TRAIN["batch_size"],
                         remove_columns=["text"])
eval_ds  = eval_raw.map(tok,
                        batched=True,
                        batch_size=TRAIN["batch_size"],
                        remove_columns=["text"])

# only convert to torch.Tensors at batch time
train_ds.set_format(type="torch",
                    columns=["input_ids", "attention_mask", "labels"])
eval_ds.set_format(type="torch",
                   columns=["input_ids", "attention_mask", "labels"])

collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# ─────────────────────────────────────────────────────────────
# 4. Trainer setup
# ─────────────────────────────────────────────────────────────
args = TrainingArguments(
    output_dir=str(OUT),
    per_device_train_batch_size=TRAIN["batch_size"],
    per_device_eval_batch_size=TRAIN["batch_size"],
    gradient_accumulation_steps=TRAIN["grad_accum"],
    learning_rate=TRAIN["lr"],
    num_train_epochs=TRAIN["epochs"],
    logging_steps=20,
    eval_steps=50,
    eval_strategy="steps",
    save_strategy="steps",
    save_steps=50,
    fp16=TRAIN["fp16"],
    warmup_ratio=0.1,
    optim="paged_adamw_8bit",
    lr_scheduler_type="linear",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=1,
)

class ClearCache(TrainerCallback):
    def on_step_end(self, *args, **kwargs):
        torch.cuda.empty_cache()

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=collator,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(1), ClearCache()],
)

# ─────────────────────────────────────────────────────────────
# 5. Train
# ─────────────────────────────────────────────────────────────
print(f"[{RUN}] training → {OUT}")
start_time = time.time()
trainer.train()
duration = time.time() - start_time
print(f"✔ Training finished in {duration/60:.1f} min")

# free rocky memory mid-run
torch.cuda.empty_cache()
gc.collect()

# ─────────────────────────────────────────────────────────────
# 6. Save final model + metrics
# ─────────────────────────────────────────────────────────────
model.save_pretrained(OUT)
tokenizer.save_pretrained(OUT)

pathlib.Path("metrics").mkdir(exist_ok=True)
json.dump(
    {"loss": trainer.state.log_history[-1].get("loss"),
     "seconds": duration,
     "run": RUN},
    open("metrics/eval_local.json", "w"),
    indent=2,
)

print("✔ Saved model + metrics")

# ─────────────────────────────────────────────────────────────
# 7. Final cleanup
# ─────────────────────────────────────────────────────────────
del trainer, model, train_ds, eval_ds, raw, tokenizer
torch.cuda.empty_cache()
gc.collect()
