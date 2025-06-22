# 🔬 Genomic QA Fine-Tuning

This repository contains a script to fine-tune a causal LLM on a custom genomic question-answer dataset.

---

## 📋 Table of Contents

1. [Overview](#-overview)
2. [Quick Start](#-quick-start)
3. [Generating `requirements.txt`](#-generating-requirementstxt)
4. [Script Structure](#-script-structure)
5. [Usage](#-usage)
6. [License](#-license)

---

## 🔍 Overview

The `finetune_genomic_qa.py` script:

* Loads a quantized causal language model from a local path
* Applies LoRA for parameter-efficient fine-tuning
* Formats a JSON Q\&A dataset into instruction-response pairs
* Masks labels before the assistant prompt
* Trains with mixed precision and 4-bit quantization
* Saves the fine-tuned model and tokenizer

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/Luv-crypto/genomic-qa-finetune.git
cd genomic-qa-finetune

# 2. Create & activate a virtual environment
python -m venv venv
source venv/bin/activate   # Windows PowerShell: .\\venv\\Scripts\\Activate.ps1

# 3. Install minimal deps (for code parsing)
pip install astunparse

# 4. Generate requirements.txt for used libraries
python extract_requirements.py finetune_genomic_qa.py

# 5. Install runtime dependencies
pip install -r requirements.txt

# 6. Run fine-tuning
python finetune_genomic_qa.py
```

---

## 📜 Generating `requirements.txt`

Use `extract_requirements.py` to extract only the third-party packages imported in your script:

```python
# extract_requirements.py
import ast
import sys
from pathlib import Path

# Usage: python extract_requirements.py <script.py>
script = Path(sys.argv[1]).read_text()

class Collector(ast.NodeVisitor):
    def __init__(self): self.mods = set()
    def visit_Import(self, node):
        for n in node.names: self.mods.add(n.name.split('.')[0])
    def visit_ImportFrom(self, node):
        if node.module: self.mods.add(node.module.split('.')[0])

# Parse and collect imports
tree = ast.parse(script)
col = Collector(); col.visit(tree)

# Filter out standard libraries
std = {"os","sys","json","time","math","typing","pathlib","datetime"}
pkgs = sorted(col.mods - std)

# Write requirements
with open('requirements.txt','w') as f:
    for pkg in pkgs: f.write(f"{pkg}\n")
print("Extracted packages:", pkgs)
```

Run:

```bash
python extract_requirements.py finetune_genomic_qa.py
```

This creates `requirements.txt` listing only the libraries your script uses.

---

## 🗂 Script Structure

* `finetune_genomic_qa.py`: Main fine-tuning script
* `extract_requirements.py`: Utility to generate `requirements.txt`
* `dataset/`: Place your `genomic_qa_dataset_5000.json` file here
* `offload/`: Temporary folder for quantization offload

---

## 🖥 Usage

Edit the configuration section at the top of `finetune_genomic_qa.py`:

```python
LOCAL_MODEL_PATH = "path/to/model"
DATASET_NAME = "dataset/genomic_qa_dataset_5000.json"
SAMPLE_SIZE = 4800
EVAL_SIZE = 200
MAX_LENGTH = 256
```

Then run the quick start steps above.

---

## 📄 License

MIT © Luv-crypto
