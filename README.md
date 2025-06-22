# 🔬 Genomic QA Fine-Tuning

This repository contains a script to fine-tune a causal LLM on a custom genomic question-answer dataset.

---

## 📋 Table of Contents

1. [Overview](#-overview)
2. [Quick Start](#-quick-start)
3. [Script Structure](#-script-structure)
4. [Usage](#-usage)
5. [License](#-license)

---

## 🔍 Overview

This repository provides a straightforward pipeline to fine-tune a causal language model on a custom genomic question-answer dataset using parameter-efficient methods. The workflow includes:

* Loading a pre-trained model from a local directory
* Applying efficient fine-tuning (e.g., LoRA)
* Preparing and formatting a JSON Q\&A dataset into instruction-response pairs
* Training with mixed precision settings for speed and resource efficiency
* Saving the resulting fine-tuned model and tokenizer

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/Luv-crypto/genomic-qa-finetune.git
cd genomic-qa-finetune

# 2. Create & activate a virtual environment
python -m venv venv
source venv/bin/activate   # Windows PowerShell: .\\venv\\Scripts\\Activate.ps1

# 3. Install runtime dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Run fine-tuning
python finetune_genomic_qa.py
```

The provided `requirements.txt` includes:

```
datasets==3.6.0
numpy==1.24.4
peft==0.15.2
torch==2.6.0+cu118
transformers==4.52.4
```

---

## 🗂 Script Structure

* `finetune_genomic_qa.py`: Main fine-tuning script
* `requirements.txt`: Pinned runtime dependencies
* `genomic_qa_dataset_5000.json`: Place your JSON Q\&A dataset file in the project root (or adjust `DATASET_NAME` in the script to its location)
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

Then follow the Quick Start steps above.

---

## 📄 License

MIT © Luv-crypto
