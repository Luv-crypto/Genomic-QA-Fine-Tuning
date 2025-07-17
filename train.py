import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    BitsAndBytesConfig
)
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import os
from transformers import TrainerCallback
import numpy as np

# ────────────────────────────────────────────────────────────────
# 0. Configuration - Optimized for Colab
# ────────────────────────────────────────────────────────────────
LOCAL_MODEL_PATH = r"C:\Users\fadial\Downloads\qwen"
DATASET_NAME = "genomic_qa_dataset_5000.json"  # Your Q&A dataset
SAMPLE_SIZE = 4800  # Training samples
EVAL_SIZE = 200    # Evaluation samples
MAX_LENGTH = 256  # Optimal for instruction-response

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# ────────────────────────────────────────────────────────────────
# 1. Model Loading
# ────────────────────────────────────────────────────────────────
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_PATH,
    local_files_only=True,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map="auto",
    offload_folder="offload",
    offload_state_dict=True
)

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# ────────────────────────────────────────────────────────────────
# 2. Tokenizer Setup - MUST BE BEFORE DATASET PROCESSING
# ────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(
    LOCAL_MODEL_PATH,
    local_files_only=True,
    trust_remote_code=True,
    use_fast=False
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ────────────────────────────────────────────────────────────────
# 3. LoRA Configuration
# ────────────────────────────────────────────────────────────────
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,  # Increased from 8 → better task adaptation
    lora_alpha=32,  # Increased from 16 → balances scale vs. regularization
    lora_dropout=0.05,  # Lowered from 0.1 → reduces over-regularization
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Added modules
    bias="lora_only",  # Changed from "none" → better fine-tuning
    modules_to_save=["lm_head"],  # Crucial addition → improves output quality
    use_rslora=True  # New → better stability & convergence
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ────────────────────────────────────────────────────────────────
# 4. Dataset Preparation & Formatting
# ────────────────────────────────────────────────────────────────
def format_qa(examples):
    """Format question-answer pairs into instruction-response format"""
    texts = []
    for q, a in zip(examples["question"], examples["answer"]):
        texts.append(f"### Human: {q}\n### Assistant: {a}")
    return {"text": texts}

# Load dataset
dataset = load_dataset("json", data_files=DATASET_NAME)

# Format into instruction-response pairs
dataset = dataset.map(
    format_qa,
    batched=True,
    remove_columns=["question", "answer"]
)

# Create subsets - fixed to avoid negative indexing
train_data = dataset["train"].select(range(SAMPLE_SIZE))
eval_data = dataset["train"].select(range(SAMPLE_SIZE, SAMPLE_SIZE+EVAL_SIZE))

# ────────────────────────────────────────────────────────────────
# 5. Tokenization with Label Masking
# ────────────────────────────────────────────────────────────────
def tokenize_function(examples):
    # Tokenize with padding
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    
    # Create labels by copying input_ids
    labels = tokenized["input_ids"].clone()
    
    # Mask instruction tokens
    for i in range(labels.shape[0]):
        input_ids = tokenized["input_ids"][i]
        
        # Find "Assistant:" token positions
        assistant_pos = None
        for idx in range(len(input_ids) - 1):
            if tokenizer.decode(input_ids[idx:idx+1]) == "Assistant":
                if tokenizer.decode(input_ids[idx:idx+8]) == "Assistant:":
                    assistant_pos = idx
                    break
        
        # Mask everything before "Assistant:"
        if assistant_pos is not None:
            labels[i, :assistant_pos+8] = -100
    
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": labels
    }

# Apply tokenization
train_dataset = train_data.map(
    tokenize_function, 
    batched=True, 
    batch_size=4,
    remove_columns=["text"]
)

eval_dataset = eval_data.map(
    tokenize_function, 
    batched=True, 
    batch_size=4,
    remove_columns=["text"]
)

# ────────────────────────────────────────────────────────────────
# 6. Data Collator
# ────────────────────────────────────────────────────────────────
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# ────────────────────────────────────────────────────────────────
# 7. Training Arguments
# ────────────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir="D://test//genomic-assistant-results",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    num_train_epochs=20,
    logging_steps=20,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=100,
    fp16=True,
    warmup_ratio=0.1,
    optim="paged_adamw_8bit",
    lr_scheduler_type="linear",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=1,
    remove_unused_columns=False,
)

# ────────────────────────────────────────────────────────────────
# 8. Memory Management Callback
# ────────────────────────────────────────────────────────────────
class ClearCacheCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()

# ────────────────────────────────────────────────────────────────
# 9. Trainer Setup
# ────────────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=1),
        ClearCacheCallback()
    ]
)

# ────────────────────────────────────────────────────────────────
# 10. Start Training
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("offload", exist_ok=True)
    print("Starting training...")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")
    
    # Start training
    trainer.train()
    
    print("Training complete!")
    
    # Save final model
    model.save_pretrained("D://test//genomic-assistant")
    tokenizer.save_pretrained("D://test//genomic-assistant")
    
    # Test with a genomic question
    prompt = "### Human: What is the equation used for attention scores in transformer architectures?\n### Assistant:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9
    )
    print("Model response:")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))