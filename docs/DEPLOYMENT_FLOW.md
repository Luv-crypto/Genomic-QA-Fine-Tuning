# Deployment Flow

This document explains the end-to-end workflow for training, evaluating and serving models using this repository.

## 1. Clone and Setup
1. Clone the repository and install dependencies from `requirements.txt`.
2. Run `dvc pull -q` to download datasets and existing model artifacts from the remote.

## 2. Configure Parameters
1. Edit `params.yaml` and set `family` and `run_name` for the model run.
2. Update dataset paths and any LoRA or training hyper-parameters.

## 3. Train and Promote
1. Run `python -m model_lab.train` or `Scripts/run_train.ps1` to start training.
2. Promote the WIP run to a candidate using `Scripts/promote.ps1 '<timestamp>-<run_name>'`.
3. Commit and push the generated `.dvc` pointer files using Git and DVC.

## 4. Reproduce Pipeline
Run `dvc repro` to execute all stages defined in `dvc.yaml`:
1. **merge** – merges LoRA adapters with the base model.
2. **gguf** – converts the merged weights to GGUF format.
3. **quant** – optionally quantizes the GGUF file.
4. **modelfile** – writes an Ollama `Modelfile` for the run.
5. **eval_ci** – evaluates the candidate model.
6. **docker** – builds a Docker image that contains the GGUF weights and Modelfile.

After completion, `.ci/image_<family>_<run_name>.txt` contains the image tag.

## 5. Serve with Ollama
1. Pull the image and start a container:
   ```bash
   docker pull $(cat .ci/image_<family>_<run_name>.txt)
   docker run -d --name <family>_<run_name> -p 11434:11434 $(cat .ci/image_<family>_<run_name>.txt)
   ```
2. Launch the A/B testing UI:
   ```bash
   python src/ui/app.py
   ```

## 6. Promote to Stable
1. Monitor feedback in `user_feedback.json` for ~24 hours.
2. If the candidate receives at least 50% positive votes, run
   `Scripts/switch_stable.ps1 <run_name>` to update `models/hf/stable/<family>.dvc`.
   Previous stable pointers are archived under `models/hf/stable/archive/` and can
   be restored if needed.

## 7. Rollbacks and Multiple Families
- To roll back, replace the stable pointer with a file from the `archive/` directory.
- Each model family has its own `stable/<family>.dvc` so families can evolve independently.
- The UI can load multiple families or versions by providing a JSON registry file via the
  `MODEL_REGISTRY_FILE` environment variable. The format is:
  ```json
  {
      "qwen_v2": {"label": "Qwen Stable", "ollama_name": "qwen_v2", "url": "http://localhost:11434/v1/chat/completions"},
      "mistral_v1": {"label": "Mistral", "ollama_name": "mistral", "url": "http://localhost:11435/v1/chat/completions"}
  }
  ```

This workflow ensures any team member can train a new model, promote it through the
pipeline and deploy it with A/B testing while keeping prior stable versions available.
