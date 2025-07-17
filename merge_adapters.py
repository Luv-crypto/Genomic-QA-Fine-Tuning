# merge_adapters.py
import os
os.environ["USE_TORCH_DTENSOR"] = "0"  # Ensure DTensor remains disabled
os.environ["SAFETENSORS_FAST_GPU"] = "1"  # Accelerate safetensors

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Configuration - UPDATE THESE PATHS
BASE_MODEL_PATH = "C:\\Users\\fadial\\Downloads\\qwen"  # Original base model
ADAPTER_PATH = "D:\\LLM_prod\\Genomic-QA-Fine-Tuning\\genomic-assistant"  # Your trained adapter
MERGED_MODEL_PATH = "D:\\LLM_prod\\Genomic-QA-Fine-Tuning\\genomic-assistant-merged"  # Output directory

# Load base model (without quantization for merging)
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# Load adapter
print("Loading adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)

# Merge adapter with base model
print("Merging adapter with base model...")
merged_model = model.merge_and_unload()

# Save merged model with safetensors
print("Saving merged model...")
merged_model.save_pretrained(
    MERGED_MODEL_PATH,
    safe_serialization=True  # Critical for proper saving
)

# Save tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
tokenizer.save_pretrained(MERGED_MODEL_PATH)

print(f"Successfully merged and saved model to {MERGED_MODEL_PATH}")

# Add this to your merge script after saving the model
import gc
import torch

# Release model references
del model
del merged_model
# Force garbage collection
gc.collect()

# Clear PyTorch cache
torch.cuda.empty_cache()



print("Memory resources released")