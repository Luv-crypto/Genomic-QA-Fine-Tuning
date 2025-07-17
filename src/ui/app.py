"""
Flask Chat UI for Ollama‑served Models with Optional Pairwise A/B Testing
=========================================================================

This is a **full, drop‑in replacement** for your earlier script. It keeps your overall UX but adds a robust
**pairwise A/B comparison mode** so users can see *two model responses side‑by‑side* **before** being asked for
feedback.

---
## Quick Start

```bash
# (recommended) create a venv first
pip install flask psutil requests

# env overrides (optional)
export AB_TEST_MODE=pair            # pair | interleaved
export AB_TEST_ENABLED=1            # 1=on, 0=off (ignored if mode=pair; pair implies A/B)
export BLIND_TEST=1                 # 1=hide which model answered, 0=show internal name
export OLLAMA_API_URL=http://localhost:11434/v1/chat/completions
export OLLAMA_QWEN_STABLE=qwen2:7b-instruct-q4_0
export OLLAMA_QWEN_EXP=qwen2:7b-instruct
export OLLAMA_QWEN_QUANT=qwen2:7b-instruct-q4_K_M
export OLLAMA_KEEP_ALIVE=30m
python app.py
```

Replace the `OLLAMA_QWEN_*` names with the exact model names from `ollama list`.

---
## Modes

**interleaved** – (old behavior, but fixed) one model per response; randomly sampled each request when A/B enabled.
**pair** – (new) for each user question the backend queries two variants, renders them side‑by‑side (Answer A / Answer B),
then prompts the user to choose a winner (or tie / neither) and optionally rate each.

---
## File Outputs
- `user_feedback.json` – Stores interleaved AND pairwise feedback (separate keys).
- `model_performance.json` – Aggregate latency + memory metrics per internal model ID.

---
## IMPORTANT – Fill These In
Scroll to the **CONFIG** block and change the `MODELS` mapping so `ollama_name` matches your installed models.

---
"""

from __future__ import annotations

import os
import re
import json
import time
import psutil
import random
import datetime as dt
from pathlib import Path
from typing import Dict, Any, Tuple, List

import requests
from flask import (
    Flask,
    render_template_string,
    request,
    session,
    jsonify,
)

# --------------------------------------------------------------------------------------
# --------------------------------- CONFIG --------------------------------------------
# --------------------------------------------------------------------------------------

# Data files
FEEDBACK_FILE = Path(os.getenv("FEEDBACK_FILE", "user_feedback.json"))
PERFORMANCE_FILE = Path(os.getenv("PERFORMANCE_FILE", "model_performance.json"))

# AB test global mode: 'pair' (side‑by‑side comparison) or 'interleaved' (one model / response).
AB_TEST_MODE = os.getenv("AB_TEST_MODE", "pair").strip().lower()
assert AB_TEST_MODE in {"pair", "interleaved"}, "AB_TEST_MODE must be 'pair' or 'interleaved'"

# Interleaved mode master switch. (Ignored when AB_TEST_MODE='pair' because pair *is* an A/B test.)
AB_TEST_ENABLED = bool(int(os.getenv("AB_TEST_ENABLED", "1")))

# Blind users to which variant served? (1=True / hide, 0=False / show badges)
BLIND_TEST = bool(int(os.getenv("BLIND_TEST", "1")))

# Keep‑alive hint to Ollama; may be string ("30m", "-1") or numeric seconds if supported by your build.
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")

# System prompt used for all model calls (short safe answer; change to taste)
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a domain assistant. Provide a clear, correct answer. Be concise but helpful.",
)

# STOP tokens (trim hallucinated delims)
STOP_TOKENS = [
    "### Human:", "<think>", "</think>", "<|im_end|>", "<|im_start|>", "Assistant:",
]

# Number of tokens to predict (adjust per model + need)
NUM_PREDICT = int(os.getenv("NUM_PREDICT", "256"))

# ----------------------- Model Registry ------------------------------------------------
# User sees a single architecture (Qwen) in dropdown. Backend decides which internal variant(s) to call.

# Default API endpoint – override per model if you run multi‑port Ollama servers.
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/v1/chat/completions")

# Read actual Ollama model names from env (update or hard‑code below)
# OLLAMA_QWEN_STABLE = os.getenv("OLLAMA_QWEN_STABLE", "qwen2:7b-instruct-q4_0")
OLLAMA_QWEN_EXP    = os.getenv("OLLAMA_QWEN_v2", "qwen_v2")
OLLAMA_QWEN_QUANT  = os.getenv("OLLAMA_QWEN_QUANT", "qwen_quant_v1")

# Internal model registry – keys are internal IDs; values hold display + API info
MODELS: Dict[str, Dict[str, Any]] = {
    # "qwen_v1": {
    #     "label": "Qwen Stable",
    #     "ollama_name": OLLAMA_QWEN_STABLE,
    #     "url": OLLAMA_API_URL,
    # },
    "qwen_v2": {
        "label": "Qwen Stable",
        "ollama_name": OLLAMA_QWEN_EXP,
        "url": OLLAMA_API_URL,
    },
    "qwen_quant_v1": {
        "label": "Qwen Quant",
        "ollama_name": OLLAMA_QWEN_QUANT,
        "url": OLLAMA_API_URL,
    },
}

# Which internal model is the baseline when A/B test disabled or in pair stable+challenger selection
DEFAULT_MODEL_ID = "qwen_v2"

# Arms + weights used in interleaved mode (per‑request sampling)
AB_TEST_ARMS = ["qwen_v2", "qwen_quant_v1"]
AB_TEST_WEIGHTS = [0.2, 0.8]  # must align length w/ arms

# Pair mode challengers (vs stable). Random one chosen each question.
AB_PAIR_CHALLENGERS = [mid for mid in AB_TEST_ARMS if mid != DEFAULT_MODEL_ID]
PAIR_RANDOMIZE_ORDER = True   # shuffle which answer is A / B

# Feedback sampling – interleaved mode only (pair always asks after answers shown)
# Probability of triggering a *pairwise* comparison instead of serving only the stable model.
#
#   If you set the env var PAIR_SAMPLE_PROB it will be used directly (0.0–1.0).
#   Otherwise we derive it from the AB_TEST_WEIGHTS by taking 1 - (stable_weight/total_weight).
#   Example: weights [0.9, 0.1] -> PAIR_SAMPLE_PROB ≈ 0.1 (1 in 10 questions shows both answers).
_pair_prob_env = os.getenv("PAIR_SAMPLE_PROB")
if _pair_prob_env is not None:
    try:
        PAIR_SAMPLE_PROB = max(0.0, min(1.0, float(_pair_prob_env)))
    except ValueError:  # fallback safe
        PAIR_SAMPLE_PROB = 0.5
else:
    try:
        _stable_idx = AB_TEST_ARMS.index(DEFAULT_MODEL_ID)
        _stable_w = AB_TEST_WEIGHTS[_stable_idx]
        _total_w = sum(AB_TEST_WEIGHTS) or 1.0
        PAIR_SAMPLE_PROB = max(0.0, min(1.0, 1.0 - (_stable_w / _total_w)))
    except Exception:  # pragma: no cover – very unlikely
        PAIR_SAMPLE_PROB = 0.5

# Feedback sampling – interleaved mode only (pair always asks after answers shown)
FEEDBACK_SAMPLE_PROB = float(os.getenv("FEEDBACK_SAMPLE_PROB", "0.25"))  # 25% of responses
FEEDBACK_COOLDOWN_HOURS = float(os.getenv("FEEDBACK_COOLDOWN_HOURS", "12"))

# --------------------------------------------------------------------------------------
# --------------------------- Data Persistence Helpers ---------------------------------
# --------------------------------------------------------------------------------------

def _default_feedback_struct() -> Dict[str, Any]:
    return {
        "interleaved": [],  # list of dicts {timestamp, model_used, rating, latency, question, response, comment}
        "pair": [],          # list of dicts {timestamp, a_model, b_model, preferred, ratingA, ratingB, ...}
    }

def _default_perf_struct() -> Dict[str, Any]:
    return {"models": {}}  # model_id -> stats


def load_json_safe(path: Path, default_func):
    try:
        with path.open("r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default_func()


def save_json_safe(path: Path, data: Dict[str, Any]):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


# Load at import time
feedback_data = load_json_safe(FEEDBACK_FILE, _default_feedback_struct)
performance_data = load_json_safe(PERFORMANCE_FILE, _default_perf_struct)


# --------------------------------------------------------------------------------------
# ------------------------------ Performance Logging -----------------------------------
# --------------------------------------------------------------------------------------

def measure_mem_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def update_performance_data(model_id: str, latency: float, mem_used_delta: float):
    models = performance_data.setdefault("models", {})
    if model_id not in models:
        models[model_id] = {
            "total_requests": 0,
            "total_latency": 0.0,
            "min_latency": latency,
            "max_latency": latency,
            "avg_memory": mem_used_delta,
            "last_updated": dt.datetime.utcnow().isoformat(),
        }
    m = models[model_id]
    m["total_requests"] += 1
    m["total_latency"] += latency
    m["min_latency"] = min(m["min_latency"], latency)
    m["max_latency"] = max(m["max_latency"], latency)
    # running avg of mem delta
    m["avg_memory"] = ((m["avg_memory"] * (m["total_requests"] - 1)) + mem_used_delta) / m["total_requests"]
    m["last_updated"] = dt.datetime.utcnow().isoformat()
    save_json_safe(PERFORMANCE_FILE, performance_data)


def summarize_perf_for_ui() -> Dict[str, Any]:
    out = {"models": {}}
    for mid, stats in performance_data.get("models", {}).items():
        total = stats.get("total_requests", 0) or 1
        avg_lat = stats.get("total_latency", 0.0) / total
        out["models"][MODELS.get(mid, {}).get("label", mid)] = {
            "total_requests": stats.get("total_requests", 0),
            "avg_latency": avg_lat,
            "min_latency": stats.get("min_latency", 0.0),
            "max_latency": stats.get("max_latency", 0.0),
            "avg_memory": stats.get("avg_memory", 0.0),
            "last_updated": stats.get("last_updated", "")
        }
    return out


# --------------------------------------------------------------------------------------
# ------------------------------ Feedback Helpers --------------------------------------
# --------------------------------------------------------------------------------------

def log_interleaved_feedback(*, timestamp: str, model_id: str, rating: int, latency: float, question: str, response: str, comment: str):
    feedback_data.setdefault("interleaved", []).append({
        "timestamp": timestamp,
        "model_used": model_id,
        "rating": rating,
        "latency": latency,
        "question": question,
        "response": response,
        "comment": comment,
    })
    save_json_safe(FEEDBACK_FILE, feedback_data)


def log_pair_feedback(*, timestamp: str, question: str, answers: List[Dict[str, Any]], preferred: str, ratingA: int|None, ratingB: int|None, comment: str):
    # answers: list of {modelId, label, text}
    a_model = next((a["modelId"] for a in answers if a["label"] == "A"), None)
    b_model = next((a["modelId"] for a in answers if a["label"] == "B"), None)
    feedback_data.setdefault("pair", []).append({
        "timestamp": timestamp,
        "question": question,
        "a_model": a_model,
        "b_model": b_model,
        "preferred": preferred,  # 'A','B','tie','neither'
        "ratingA": ratingA,
        "ratingB": ratingB,
        "comment": comment,
    })
    save_json_safe(FEEDBACK_FILE, feedback_data)


# --------------------------------------------------------------------------------------
# ---------------------------- Model Invocation Helpers --------------------------------
# --------------------------------------------------------------------------------------

def _ollama_payload(model_name: str, user_msg: str) -> Dict[str, Any]:
    return {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {
            "temperature": 0.1,
            "num_ctx": 512,
            "stop": STOP_TOKENS,
            "num_predict": NUM_PREDICT,
        },
    }


def call_model(model_id: str, user_msg: str) -> Tuple[str, float, float]:
    """Call a specific internal model_id. Return (cleaned_text, latency_s, mem_delta_mb)."""
    m = MODELS[model_id]
    payload = _ollama_payload(m["ollama_name"], user_msg)
    mem_before = measure_mem_mb()
    start = time.time()
    r = requests.post(m["url"], json=payload, timeout=60)
    r.raise_for_status()
    resp = r.json()
    raw_text = resp["choices"][0]["message"]["content"]
    latency = time.time() - start
    mem_after = measure_mem_mb()
    cleaned = clean_response(raw_text)
    mem_delta = max(0.0, mem_after - mem_before)
    update_performance_data(model_id, latency, mem_delta)
    return cleaned, latency, mem_delta


def choose_variant_interleaved() -> str:
    if not AB_TEST_ENABLED:
        return DEFAULT_MODEL_ID
    return random.choices(AB_TEST_ARMS, weights=AB_TEST_WEIGHTS, k=1)[0]


def choose_pair() -> Tuple[str, str]:
    """Return (stable, challenger) internal IDs."""
    if not AB_PAIR_CHALLENGERS:
        return DEFAULT_MODEL_ID, DEFAULT_MODEL_ID
    challenger = random.choice(AB_PAIR_CHALLENGERS)
    return DEFAULT_MODEL_ID, challenger


# --------------------------------------------------------------------------------------
# ------------------------------- Response Cleaning ------------------------------------
# --------------------------------------------------------------------------------------

def clean_response(text: str) -> str:
    # kill think tags
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # remove leading role prefixes
    text = re.sub(r"^(Assistant:|Model:|AI:)", "", text).strip()
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --------------------------------------------------------------------------------------
# --------------------------- HTML TEMPLATE (Jinja‑style) ------------------------------
# --------------------------------------------------------------------------------------

BULMA = "https://cdn.jsdelivr.net/npm/bulma@0.9.4/css/bulma.min.css"
HTMX = "https://unpkg.com/htmx.org@1.9.12"

HTML = """<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>Genomic-QA Chat</title>
  <link rel=\"stylesheet\" href=\"{{ bulma_url }}\">
  <script>window.AB_TEST_MODE='{{ ab_test_mode }}';window.BLIND_TEST={{ blind_test|lower }};</script>
  <style>
    .message-container{max-height:60vh;overflow-y:auto;margin-bottom:20px;}
    .thinking{text-align:center;padding:10px;display:none;}
    .answer-highlight{background:#f5f5f5;border-left:3px solid #3273dc;padding:10px;margin:5px 0;}
    .perf-col{font-size:0.8rem;color:#666;margin-top:4px;}
    .feedback-modal,.pair-feedback-modal{display:none;position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);background:white;padding:20px;border-radius:5px;box-shadow:0 0 10px rgba(0,0,0,.1);z-index:1000;width:90%;max-width:520px;}
    .overlay{display:none;position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,.5);z-index:999;}
  </style>
</head>
<body>
  <div class=\"overlay\" id=\"overlay\"></div>

  <!-- Interleaved Feedback Modal -->
  <div class=\"feedback-modal\" id=\"feedback-modal\">
    <div class=\"box\">
      <h3 class=\"title is-4\">Help us improve!</h3>
      <p>How would you rate this response?</p>
      <div class=\"buttons\">
        <button class=\"button is-success feedback-btn\" data-rating=\"5\">Excellent</button>
        <button class=\"button is-info feedback-btn\" data-rating=\"4\">Good</button>
        <button class=\"button is-warning feedback-btn\" data-rating=\"3\">Average</button>
        <button class=\"button is-danger feedback-btn\" data-rating=\"1\">Poor</button>
      </div>
      <div class=\"field\"><label class=\"label\">Comments (optional)</label><textarea class=\"textarea\" id=\"feedback-comment\"></textarea></div>
      <div class=\"field is-grouped\"><div class=\"control\"><button class=\"button is-link\" id=\"submit-feedback\">Submit</button></div><div class=\"control\"><button class=\"button is-light\" id=\"skip-feedback\">Skip</button></div></div>
    </div>
  </div>

  <!-- Pairwise Feedback Modal -->
  <div class=\"pair-feedback-modal\" id=\"pair-feedback-modal\">
    <div class=\"box\">
      <h3 class=\"title is-4\">Which answer do you prefer?</h3>
      <div id=\"pair-question\" class=\"content\"></div>
      <div class=\"field\">
        <label class=\"label\">Select one:</label>
        <div class=\"control\">
          <label class=\"radio\"><input type=\"radio\" name=\"pair-choice\" value=\"A\"> Answer A</label>
          <label class=\"radio\" style=\"margin-left:1em;\"><input type=\"radio\" name=\"pair-choice\" value=\"B\"> Answer B</label>
          <label class=\"radio\" style=\"margin-left:1em;\"><input type=\"radio\" name=\"pair-choice\" value=\"tie\"> Tie</label>
          <label class=\"radio\" style=\"margin-left:1em;\"><input type=\"radio\" name=\"pair-choice\" value=\"neither\"> Neither</label>
        </div>
      </div>
      <div class=\"field is-horizontal\">
        <div class=\"field-body\">
          <div class=\"field\"><label class=\"label\">Rate A (1-5)</label><input class=\"input\" type=\"number\" min=1 max=5 id=\"pair-rate-a\"></div>
          <div class=\"field\"><label class=\"label\">Rate B (1-5)</label><input class=\"input\" type=\"number\" min=1 max=5 id=\"pair-rate-b\"></div>
        </div>
      </div>
      <div class=\"field\"><label class=\"label\">Comments</label><textarea class=\"textarea\" id=\"pair-comment\"></textarea></div>
      <div class=\"field is-grouped\"><div class=\"control\"><button class=\"button is-link\" id=\"pair-feedback-submit\">Submit</button></div><div class=\"control\"><button class=\"button is-light\" id=\"pair-feedback-skip\">Skip</button></div></div>
    </div>
  </div>

  <section class=\"section\">
    <div class=\"container\">
      <h1 class=\"title\">Genomic Research Assistant</h1>

      <div class=\"message-container\" id=\"chat-history\"></div>
      <div class=\"thinking\" id=\"thinking\"><progress class=\"progress is-small is-primary\" max=100>Thinking...</progress><p>Processing your query...</p></div>
      <div id=\"error\" class=\"notification is-danger\" style=\"display:none;\"></div>

      <form id=\"chat-form\" class=\"box\">
        <div class=\"field\">
          <label class=\"label\">Ask a genomics question:</label>
          <div class=\"control\"><textarea class=\"textarea\" id=\"question\" name=\"question\" placeholder=\"What is CRISPR gene editing?\" required></textarea></div>
        </div>
        <div class=\"field is-grouped\">
          <div class=\"control\"><div class=\"select\"><select id=\"model-select\" name=\"model\">{{ model_options|safe }}</select></div></div>
          <div class=\"control\"><button type=\"submit\" class=\"button is-link\">Ask</button></div>
          <div class=\"control\"><button type=\"button\" class=\"button is-light\" id=\"clear-btn\">Clear Chat</button></div>
          <div class=\"control\"><button type=\"button\" class=\"button is-info\" id=\"performance-btn\">Show Performance</button></div>
        </div>
      </form>

      <div id=\"performance-table\" style=\"display:none;margin-top:20px;\">
        <h2 class=\"subtitle\">Model Performance Metrics</h2>
        <table class=\"table is-fullwidth is-striped\">
          <thead><tr><th>Model</th><th>Requests</th><th>Avg Latency</th><th>Min</th><th>Max</th><th>Avg Mem Δ</th><th>Last Updated</th></tr></thead>
          <tbody id=\"performance-data\"></tbody>
        </table>
      </div>
    </div>
  </section>

  <script src=\"{{ htmx_url }}\"></script>
  <script>
  document.addEventListener('DOMContentLoaded',function(){
    const chatForm=document.getElementById('chat-form');
    const chatHistory=document.getElementById('chat-history');
    const thinking=document.getElementById('thinking');
    const errorDiv=document.getElementById('error');
    const clearBtn=document.getElementById('clear-btn');
    const performanceBtn=document.getElementById('performance-btn');
    const performanceTable=document.getElementById('performance-table');
    const performanceData=document.getElementById('performance-data');

    const overlay=document.getElementById('overlay');
    const fbModal=document.getElementById('feedback-modal');
    const pairModal=document.getElementById('pair-feedback-modal');
    const fbComment=document.getElementById('feedback-comment');
    const fbSubmit=document.getElementById('submit-feedback');
    const fbSkip=document.getElementById('skip-feedback');

    const pairSubmit=document.getElementById('pair-feedback-submit');
    const pairSkip=document.getElementById('pair-feedback-skip');
    const pairQuestionDiv=document.getElementById('pair-question');

    // restore history
    const saved=localStorage.getItem('chat-history');
    if(saved){chatHistory.innerHTML=saved;chatHistory.scrollTop=chatHistory.scrollHeight;}

    // performance toggle
    performanceBtn.addEventListener('click',()=>{
      if(performanceTable.style.display==='none'){loadPerformance();performanceTable.style.display='block';performanceBtn.textContent='Hide Performance';}
      else{performanceTable.style.display='none';performanceBtn.textContent='Show Performance';}
    });

    function loadPerformance(){fetch('/performance').then(r=>r.json()).then(data=>{performanceData.innerHTML='';for(const[model,stats]of Object.entries(data.models)){const tr=document.createElement('tr');tr.innerHTML=`<td>${model}</td><td>${stats.total_requests}</td><td>${stats.avg_latency.toFixed(2)}s</td><td>${stats.min_latency.toFixed(2)}s</td><td>${stats.max_latency.toFixed(2)}s</td><td>${stats.avg_memory.toFixed(2)}MB</td><td>${stats.last_updated}</td>`;performanceData.appendChild(tr);}});}

    // interleaved feedback gating ------------------------------------------------------
    function shouldAskFeedback(){if(window.AB_TEST_MODE!=='interleaved')return false;const last=localStorage.getItem('lastFeedbackAsked');if(!last)return Math.random()<{{ feedback_sample_prob }};const hrs=(Date.now()-Date.parse(last))/36e5;if(hrs>{{ feedback_cooldown_hours }})return Math.random()<{{ feedback_sample_prob }};return false;}

    function openInterleavedFB(modelId,question,responseText,latency){currentFB={modelId,question,responseText,latency,timestamp:new Date().toISOString()};overlay.style.display='block';fbModal.style.display='block';}
    function closeInterleavedFB(){overlay.style.display='none';fbModal.style.display='none';fbComment.value='';localStorage.setItem('lastFeedbackAsked',new Date().toISOString());currentFB=null;}

    document.querySelectorAll('.feedback-btn').forEach(btn=>{btn.addEventListener('click',()=>{const rating=parseInt(btn.dataset.rating);submitInterleavedFB(rating);});});
    fbSubmit.addEventListener('click',()=>submitInterleavedFB(3));
    fbSkip.addEventListener('click',closeInterleavedFB);

    function submitInterleavedFB(rating){if(!currentFB)return;fetch('/feedback',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...currentFB,rating:rating,comment:fbComment.value})});closeInterleavedFB();}

    // pair feedback --------------------------------------------------------------------
    let currentPair=null; // {question,answers:[{modelId,label,text}],timestamp}

    window.openPairFeedback=function(btn){const box=btn.closest('.ab-pair-box');const q=box.dataset.question;const answers=[...box.querySelectorAll('[data-model-id]')].map(n=>({modelId:n.dataset.modelId,label:n.dataset.label,text:n.querySelector('.answer-highlight').textContent}));currentPair={question:q,answers:answers,timestamp:new Date().toISOString()};pairQuestionDiv.textContent=q;overlay.style.display='block';pairModal.style.display='block';};

    function closePairFB(){overlay.style.display='none';pairModal.style.display='none';document.querySelectorAll('input[name="pair-choice"]').forEach(r=>r.checked=false);document.getElementById('pair-rate-a').value='';document.getElementById('pair-rate-b').value='';document.getElementById('pair-comment').value='';currentPair=null;}
    pairSkip.addEventListener('click',closePairFB);

    pairSubmit.addEventListener('click',()=>{if(!currentPair)return;let pref='neither';document.querySelectorAll('input[name="pair-choice"]').forEach(r=>{if(r.checked)pref=r.value;});const ratingA=parseInt(document.getElementById('pair-rate-a').value)||null;const ratingB=parseInt(document.getElementById('pair-rate-b').value)||null;const comment=document.getElementById('pair-comment').value;fetch('/feedback_pair',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({timestamp:currentPair.timestamp,question:currentPair.question,answers:currentPair.answers,preferred:pref,ratingA:ratingA,ratingB:ratingB,comment:comment})});closePairFB();});

    // form submit ----------------------------------------------------------------------
    chatForm.addEventListener('submit',async function(e){e.preventDefault();const question=document.getElementById('question').value.trim();if(!question)return;const model=document.getElementById('model-select').value;errorDiv.style.display='none';errorDiv.textContent='';thinking.style.display='block';addUserMessage(question);document.getElementById('question').value='';let ep='/chat';if(window.AB_TEST_MODE==='pair')ep='/chat_pair';try{const resp=await fetch(ep,{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:new URLSearchParams({question:question,model:model})});if(!resp.ok)throw new Error('API error '+resp.status);const html=await resp.text();chatHistory.innerHTML+=html;chatHistory.scrollTop=chatHistory.scrollHeight;localStorage.setItem('chat-history',chatHistory.innerHTML);if(window.AB_TEST_MODE==='interleaved'){// extract last message meta
  const last=chatHistory.lastElementChild;const modelId=last?.dataset?.modelId||'unknown';const latency=parseFloat(last?.dataset?.latency||'0');const respText=last?last.querySelector('.answer-highlight').textContent:'';if(shouldAskFeedback())openInterleavedFB(modelId,question,respText,latency);
}
    }catch(err){errorDiv.textContent=err.message;errorDiv.style.display='block';console.error(err);}finally{thinking.style.display='none';}});

    clearBtn.addEventListener('click',()=>{chatHistory.innerHTML='';localStorage.removeItem('chat-history');});

    function addUserMessage(content){const div=document.createElement('div');div.className='message is-info';div.innerHTML=`<div class="message-header">You</div><div class="message-body">${content}</div>`;chatHistory.appendChild(div);}
  });
  </script>
</body></nhtml>"""

# --------------------------------------------------------------------------------------
# ------------------------------ Flask App ---------------------------------------------
# --------------------------------------------------------------------------------------

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = False  # change if behind TLS


# ------------------------------ Routes -------------------------------------------------

@app.get('/')
def index():
    # Only one user-facing architecture (Qwen)
    model_options = '<option value="qwen">Qwen</option>'
    return render_template_string(
        HTML,
        bulma_url=BULMA,
        htmx_url=HTMX,
        ab_test_mode=AB_TEST_MODE,
        blind_test=str(BLIND_TEST).lower(),
        model_options=model_options,
        feedback_sample_prob=FEEDBACK_SAMPLE_PROB,
        feedback_cooldown_hours=FEEDBACK_COOLDOWN_HOURS,
    )


# ----- interleaved (single-answer) endpoint --------------------------------------------
@app.post('/chat')
def chat_interleaved():
    user_msg = request.form['question'].strip()
    # architecture user choice ignored (single)
    model_id = choose_variant_interleaved()
    try:
        cleaned, latency, _mem = call_model(model_id, user_msg)
        html = render_interleaved_html(cleaned, latency, model_id)
        return html
    except Exception as e:
        return render_error_html(str(e))


# ----- pairwise endpoint ----------------------------------------------------------------
@app.post('/chat_pair')
def chat_pair():
    """Handle chat when AB_TEST_MODE='pair'.

    Behavior controlled by PAIR_SAMPLE_PROB:
      • With probability PAIR_SAMPLE_PROB -> run *pairwise* stable+challenger comparison.
      • Otherwise -> return *single* stable answer (no feedback button shown).

    This lets you express coarse traffic allocation using weights, e.g.,
    AB_TEST_WEIGHTS = [0.9, 0.1]  => ~10% of user questions trigger a side‑by‑side compare.
    """
    user_msg = request.form['question'].strip()

    # Always serve stable only if A/B disabled globally
    if not AB_TEST_ENABLED:
        try:
            cleaned, latency, _ = call_model(DEFAULT_MODEL_ID, user_msg)
            return render_interleaved_html(cleaned, latency, DEFAULT_MODEL_ID)
        except Exception as e:  # noqa
            return render_error_html(str(e))

    # Sample against PAIR_SAMPLE_PROB gate -----------------------------------------
    if random.random() > PAIR_SAMPLE_PROB:
        # Serve stable only
        try:
            cleaned, latency, _ = call_model(DEFAULT_MODEL_ID, user_msg)
            return render_interleaved_html(cleaned, latency, DEFAULT_MODEL_ID)
        except Exception as e:  # noqa
            return render_error_html(str(e))

    # Otherwise: run full pair compare ----------------------------------------------
    stable, chall = choose_pair()
    try:
        text_a, lat_a, _ = call_model(stable, user_msg)
        text_b, lat_b, _ = call_model(chall, user_msg)
        answers = [
            {"id": stable, "label": "A", "content": text_a, "latency": lat_a},
            {"id": chall,  "label": "B", "content": text_b, "latency": lat_b},
        ]
        if PAIR_RANDOMIZE_ORDER:
            random.shuffle(answers)
            for i, a in enumerate(answers):
                a['label'] = 'A' if i == 0 else 'B'
        html = render_pair_html(user_msg, answers)
        return html
    except Exception as e:  # noqa
        return render_error_html(str(e))


# ----- performance ---------------------------------------------------------------------
@app.get('/performance')
def get_performance():
    return jsonify(summarize_perf_for_ui())


# ----- feedback (interleaved) ----------------------------------------------------------
@app.post('/feedback')
def handle_feedback_interleaved():
    data = request.json
    try:
        log_interleaved_feedback(
            timestamp=data['timestamp'],
            model_id=data['modelId'],
            rating=data.get('rating', 3),
            latency=data.get('latency', 0.0),
            question=data.get('question', ''),
            response=data.get('response', ''),
            comment=data.get('comment', ''),
        )
        return {"status": "success"}
    except Exception as e:  # noqa
        return {"status": "error", "message": str(e)}, 500


# ----- feedback (pairwise) -------------------------------------------------------------
@app.post('/feedback_pair')
def handle_feedback_pair():
    data = request.json
    try:
        log_pair_feedback(
            timestamp=data['timestamp'],
            question=data.get('question', ''),
            answers=data.get('answers', []),
            preferred=data.get('preferred', 'neither'),
            ratingA=data.get('ratingA'),
            ratingB=data.get('ratingB'),
            comment=data.get('comment', ''),
        )
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}, 500


# --------------------------------------------------------------------------------------
# ------------------------------- HTML Renderers ---------------------------------------
# --------------------------------------------------------------------------------------

def render_error_html(msg: str) -> str:
    return f"""
    <div class='notification is-danger'>Error: {msg}</div>
    """


def render_interleaved_html(cleaned: str, latency: float, model_id: str) -> str:
    badge = "" if BLIND_TEST else f"<span class='tag is-info is-light'>{MODELS[model_id]['label']}</span>"
    return f"""
    <div class='message is-success' data-model-id='{model_id}' data-latency='{latency:.4f}'>
      <div class='message-header'>Genomic Assistant</div>
      <div class='message-body'>
        <div class='answer-highlight'>{cleaned}</div>
        <div class='perf-col'>Response time: {latency:.2f}s {badge}</div>
      </div>
    </div>
    """


def render_pair_html(question: str, answers: List[Dict[str, Any]]) -> str:
    # answers: list of {id,label,content,latency}
    cols = []
    for ans in answers:
        badge = "" if BLIND_TEST else f"<span class='tag is-info is-light'>{MODELS[ans['id']]['label']}</span>"
        cols.append(f"""
        <div class='column'>
          <article class='message is-success' data-model-id='{ans['id']}' data-label='{ans['label']}'>
            <div class='message-header'>Answer {ans['label']}</div>
            <div class='message-body'>
              <div class='answer-highlight'>{ans['content']}</div>
              <div class='perf-col'>Response time: {ans['latency']:.2f}s {badge}</div>
            </div>
          </article>
        </div>
        """)
    compare_btn = """
    <div class='has-text-centered' style='margin-top:1rem;'>
      <button class='button is-primary' onclick='openPairFeedback(this)'>Rate these Answers</button>
    </div>"""
    return f"""
    <div class='ab-pair-box box' data-question="{question}">
      <h2 class='title is-5'>Your Question</h2>
      <p>{question}</p>
      <hr/>
      <div class='columns is-variable is-3'>
        {''.join(cols)}
      </div>
      {compare_btn}
    </div>
    """


# --------------------------------------------------------------------------------------
# --------------------------------- Main -----------------------------------------------
# --------------------------------------------------------------------------------------
if __name__ == '__main__':
    # Ensure data files exist
    if not FEEDBACK_FILE.exists():
        save_json_safe(FEEDBACK_FILE, _default_feedback_struct())
    if not PERFORMANCE_FILE.exists():
        save_json_safe(PERFORMANCE_FILE, _default_perf_struct())

    port = int(os.getenv('PORT', '5000'))
    app.run(host='0.0.0.0', port=port, debug=True)
