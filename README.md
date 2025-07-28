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
* Applying efficient fine-tuning 
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

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install "dvc[s3]" yq powershell-yaml

# 4. Pull datasets with DVC
dvc pull -q

# 5. Adjust params.yaml (family, run_name, dataset paths)

# 6. Train
python -m model_lab.train

# 7. (Optional) quick evaluation
pwsh Scripts/run_temp_eval.ps1

# 8. Promote the run and push pointers
pwsh Scripts/promote.ps1 '<timestamp>-<run_name>'
dvc push
git push

# 9. Reproduce the full pipeline locally
dvc repro

# 10. Serve the resulting image with Ollama and launch the UI
docker pull $(cat .ci/image_*_${run_name}.txt)
docker run -d --name ${run_name} -p 11434:11434 $(cat .ci/image_*_${run_name}.txt)
python src/ui/app.py

# Optionally spin up UI + models via docker-compose
docker compose up
```
---

## 🗂 Repo Structure

* `Finetuned_qwen.ipynb`: Main fine-tuning file
* `requirements.txt`: Pinned runtime dependencies
* `genomic_qa_dataset_5000.json`: Place your JSON Q\&A dataset file in the project root (or adjust `DATASET_NAME` in the script to its location)

---


## 📁 Model Directory Layout

Artifacts are organised by **family** so teams can train and roll back versions independently.

```
models/
  hf/
    wip/<family>/<timestamp>-<run_name>/      # raw training outputs
    candidates/<family>/<run_name>_merged/    # merged checkpoints
    stable/<family>.dvc                       # pointer to active stable run
    stable/archive/<family>/<timestamp>.dvc   # previous stable pointers
  gguf/<family>/                              # converted/quantised weights
  modelfiles/<family>/<run_name>.Modelfile    # Ollama configs
```

Set `family` and `run_name` in `params.yaml` before each run.

## 🔄 Pipeline Workflow

1. **Train** – run `python -m model_lab.train` or `Scripts/run_train.ps1`.
   Outputs go to `models/hf/wip/<family>/<timestamp>-<run_name>`.
2. **Promote** – call `Scripts/promote.ps1 '<timestamp>-<run_name>'` to copy the
   run to `models/hf/candidates/<family>` and commit the DVC pointer.
3. **Reproduce** – execute `dvc repro` to merge adapters, convert to GGUF,
   optionally quantise, generate a Modelfile and build the Docker image.
4. **Serve & Test** – pull the image, start the container with Ollama, then run
   `src/ui/app.py` for A/B testing.
5. **Set Stable** – after ~24 h, tally votes with `python Scripts/ab_tally.py <candidate_id>`.
   If the candidate reached 50% preference, run `Scripts/switch_stable.ps1 <run_name>` to
   update `models/hf/stable/<family>.dvc`. The previous stable pointer is archived
   under `stable/archive/` so you can roll back at any time.

## 🖥 Usage

Edit the configuration section at the top of `Finetuned_qwen.ipynb`:

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

For a detailed pipeline walkthrough see [docs/DEPLOYMENT_FLOW.md](docs/DEPLOYMENT_FLOW.md).

