from __future__ import annotations

import base64
import glob
import os
import time
import uuid
from pathlib import Path
from threading import Event
from typing import Any, Dict, List, Tuple

import chromadb
import requests
from numpy import dot as _dot
from numpy.linalg import norm

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode

# ============================================================
# PATHS / STORAGE
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHROMA_DIR = PROJECT_ROOT / "chroma_rag"
IMG_DIR = PROJECT_ROOT / "object_store" / "images"
TBL_DIR = PROJECT_ROOT / "object_store" / "tables"

IMG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# CHROMA
# ============================================================
client = chromadb.PersistentClient(path=str(CHROMA_DIR))

txt_col = client.get_or_create_collection(
    "text_chunks",
    metadata={"hnsw:space": "cosine"}
)
img_col = client.get_or_create_collection(
    "image_chunks",
    metadata={"hnsw:space": "cosine"}
)
tbl_col = client.get_or_create_collection(
    "table_chunks",
    metadata={"hnsw:space": "cosine"}
)

# ============================================================
# DOCLING
# ============================================================
pipe_opts = PdfPipelineOptions(
    do_table_structure=True,
    generate_page_images=True,
    generate_picture_images=True,
    save_picture_images=True,
)

models_path = os.getenv("DOCLING_MODELS_PATH", "").strip()
if models_path and Path(models_path).exists():
    pipe_opts.artifacts_path = models_path

pipe_opts.table_structure_options.mode = TableFormerMode.ACCURATE

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipe_opts)
    }
)

# ============================================================
# OLLAMA / MODEL CONFIG
# ============================================================
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434").rstrip("/")
GENERATE_MODEL = os.getenv("OLLAMA_GENERATE_MODEL", "ibm:v7")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "embed:v1")

REQUEST_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "300"))


# ============================================================
# HELPERS
# ============================================================
def _post_json(url: str, payload: Dict[str, Any], retry: int = 3) -> Dict[str, Any]:
    last_err = None
    for i in range(retry):
        try:
            r = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            if i == retry - 1:
                raise
            time.sleep(1 + i)
    raise RuntimeError(f"Request failed: {last_err}")


def _chat(prompt: str, retry: int = 3) -> str:
    payload = {
        "model": GENERATE_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_ctx": 4096,
        },
    }
    data = _post_json(f"{OLLAMA_BASE}/api/generate", payload, retry=retry)
    return (data.get("response") or "").strip()


def _embed(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []

    payload = {
        "model": EMBED_MODEL,
        "input": texts,
    }
    data = _post_json(f"{OLLAMA_BASE}/api/embed", payload, retry=2)

    if "embeddings" in data and isinstance(data["embeddings"], list):
        return data["embeddings"]

    if "embedding" in data and isinstance(data["embedding"], list):
        return [data["embedding"]]

    raise RuntimeError(f"Unexpected embedding response: {data}")


def _summarize_image(path: str, caption: str = "") -> str:
    try:
        with open(path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        prompt = (
            "Summarize this image for retrieval in under 120 words. "
            "Mention charts, axes, tables, diagrams, labels, or visual findings if visible. "
            f"Caption: {caption or 'N/A'}"
        )

        payload = {
            "model": GENERATE_MODEL,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_ctx": 4096,
            },
        }
        data = _post_json(f"{OLLAMA_BASE}/api/generate", payload, retry=2)
        text = (data.get("response") or "").strip()
        return text or (caption if caption else "Relevant image from the document.")
    except Exception:
        return caption if caption else "Relevant image from the document."


def _summarize_table(table_md: str, caption: str = "") -> str:
    prompt = f"""
Summarize this table for retrieval in under 120 words.
Mention what the table contains and any key values or comparisons if clearly visible.

Caption: {caption or "N/A"}

Table:
{table_md[:4000]}
"""
    try:
        text = _chat(prompt)
        return text or (caption if caption else "Relevant table from the document.")
    except Exception:
        return caption if caption else "Relevant table from the document."


def _zip_meta(res) -> List[Dict]:
    if not res or not res.get("ids"):
        return []

    ids_raw = res["ids"][0] if isinstance(res["ids"][0], list) else res["ids"]
    metas_raw = res["metadatas"][0] if isinstance(res["metadatas"][0], list) else res["metadatas"]

    out: List[Dict] = []
    for _id, meta in zip(ids_raw, metas_raw):
        if meta:
            d = dict(meta)
            d["id"] = _id
            out.append(d)
    return out


def _cosine_top(
    question_vec: List[float],
    items: Dict[str, Dict],
    top_n: int,
    min_score: float = 0.18
) -> List[str]:
    if not items:
        return []

    ids = list(items.keys())
    summaries = [items[i].get("summary", "") for i in ids]
    if not any(summaries):
        return []

    vecs = _embed(summaries)
    sims = [_dot(question_vec, v) / (norm(question_vec) * norm(v) + 1e-9) for v in vecs]
    ranked = sorted(zip(ids, sims), key=lambda x: x[1], reverse=True)
    return [i for i, s in ranked if s >= min_score][:top_n]


def _fetch_linked_media(chunk_ids: List[str], user_id: int) -> Tuple[List[Dict], List[Dict]]:
    if not chunk_ids:
        return [], []

    where_clause = {
        "$and": [
            {"user_id": user_id},
            {
                "$or": [
                    {"parent_chunk_id": {"$in": chunk_ids}},
                    {"chunk_id": {"$in": chunk_ids}},
                ]
            },
        ]
    }

    try:
        imgs = _zip_meta(img_col.get(where=where_clause, include=["metadatas"]))
        tbls = _zip_meta(tbl_col.get(where=where_clause, include=["metadatas"]))
        return imgs, tbls
    except Exception:
        fallback = {"$and": [{"user_id": user_id}, {"parent_chunk_id": {"$in": chunk_ids}}]}
        imgs = _zip_meta(img_col.get(where=fallback, include=["metadatas"]))
        tbls = _zip_meta(tbl_col.get(where=fallback, include=["metadatas"]))
        return imgs, tbls


def _resolve_pdfs(pattern: str) -> List[Path]:
    p = Path(pattern)
    if p.exists() and p.is_file() and p.suffix.lower() == ".pdf":
        return [p.resolve()]
    return [Path(x).resolve() for x in glob.glob(pattern, recursive=True) if x.lower().endswith(".pdf")]


def _purge_existing_file_records(user_id: int, file_path: str) -> None:
    where_clause = {
        "$and": [
            {"user_id": user_id},
            {"file_path": file_path},
        ]
    }
    for col in (txt_col, img_col, tbl_col):
        try:
            col.delete(where=where_clause)
        except Exception:
            pass


# ============================================================
# INGESTION
# ============================================================
def ingest_documents(
    pattern: str,
    user_id: int,
    chunk_size: int = 1500,
    stop_event: Event | None = None
) -> None:
    stop_event = stop_event or Event()

    pdfs = _resolve_pdfs(pattern)
    if not pdfs:
        raise FileNotFoundError(f"No PDFs matched: {pattern}")

    for pdf in pdfs:
        if stop_event.is_set():
            print("Ingestion cancelled.")
            return

        print(f"Processing {pdf.name} ...")
        _purge_existing_file_records(user_id, str(pdf))

        ddoc = converter.convert(pdf).document
        md = ddoc.export_to_markdown() or ""
        if not md.strip():
            continue

        file_meta = {
            "user_id": user_id,
            "file_name": pdf.name,
            "file_path": str(pdf),
        }

        # ---------------- text chunks ----------------
        chunks = [md[i:i + chunk_size] for i in range(0, len(md), chunk_size)]
        chunk_ids = [str(uuid.uuid4()) for _ in chunks]

        for cid, chunk in zip(chunk_ids, chunks):
            txt_col.add(
                ids=[cid],
                embeddings=[_embed([chunk])[0]],
                documents=[chunk],
                metadatas=[{
                    **file_meta,
                    "chunk_id": cid,
                    "chunk_preview": chunk[:300],
                }]
            )

        # ---------------- images ----------------
        max_pg = max([pic.prov[0].page_no for pic in ddoc.pictures if pic.prov] + [1])

        for pic in ddoc.pictures:
            img = pic.get_image(ddoc)
            if img is None:
                continue

            pg = pic.prov[0].page_no if pic.prov else 1
            idx = min(int((pg - 1) / max_pg * len(chunk_ids)), len(chunk_ids) - 1)
            parent_chunk_id = chunk_ids[idx]

            img_id = str(uuid.uuid4())
            img_path = (IMG_DIR / f"{img_id}_{pdf.stem}_p{pg}.png").resolve()
            img.save(img_path, "PNG")

            caption = (pic.caption_text(ddoc) or "").strip()
            summary = _summarize_image(str(img_path), caption)
            emb_text = f"{caption}\n\n{summary}" if caption else summary

            img_col.add(
                ids=[img_id],
                embeddings=[_embed([emb_text])[0]],
                documents=[summary],
                metadatas=[{
                    **file_meta,
                    "parent_chunk_id": parent_chunk_id,
                    "path": str(img_path),
                    "caption": caption,
                    "summary": summary,
                }]
            )

        # ---------------- tables ----------------
        max_pg_tbl = max([t.prov[0].page_no for t in ddoc.tables if t.prov] + [1])

        for tbl in ddoc.tables:
            table_md = (tbl.export_to_markdown(ddoc) or "").strip()
            if not table_md:
                continue

            pg = tbl.prov[0].page_no if tbl.prov else 1
            idx = min(int((pg - 1) / max_pg_tbl * len(chunk_ids)), len(chunk_ids) - 1)
            parent_chunk_id = chunk_ids[idx]

            tbl_id = str(uuid.uuid4())
            tbl_path = (TBL_DIR / f"{tbl_id}.md").resolve()
            tbl_path.write_text(table_md, encoding="utf-8")

            caption = (tbl.caption_text(ddoc) or "").strip()
            summary = _summarize_table(table_md, caption)
            emb_text = f"{caption}\n\n{summary}" if caption else summary

            tbl_col.add(
                ids=[tbl_id],
                embeddings=[_embed([emb_text])[0]],
                documents=[summary],
                metadatas=[{
                    **file_meta,
                    "parent_chunk_id": parent_chunk_id,
                    "path": str(tbl_path),
                    "caption": caption,
                    "summary": summary,
                }]
            )


# ============================================================
# QUERY
# ============================================================
def smart_query(
    question: str,
    user_id: int,
    top_k: int = 3,
    return_media: bool = False
) -> str | Tuple[str, List[Tuple[str, str]]]:
    q_vec = _embed([question])[0]

    # ---------------- text retrieval ----------------
    txt_hits = txt_col.query(
        query_embeddings=[q_vec],
        n_results=top_k,
        where={"user_id": user_id},
        include=["documents", "metadatas"]
    )

    if not txt_hits["ids"] or not txt_hits["ids"][0]:
        answer = "Sorry, The text does not contain information about your question"
        return (answer, []) if return_media else answer

    docs = txt_hits["documents"][0]
    metas = txt_hits["metadatas"][0]
    chunk_ids = [m["chunk_id"] for m in metas]

    # ---------------- linked media ----------------
    imgs_link, tbls_link = _fetch_linked_media(chunk_ids, user_id)

    # ---------------- semantic media fallback ----------------
    imgs_sem = _zip_meta(img_col.query(
        query_embeddings=[q_vec],
        n_results=2,
        where={"user_id": user_id},
        include=["metadatas"]
    ))
    tbls_sem = _zip_meta(tbl_col.query(
        query_embeddings=[q_vec],
        n_results=2,
        where={"user_id": user_id},
        include=["metadatas"]
    ))

    imgs_all = {m["id"]: m for m in (imgs_link + imgs_sem)}
    tbls_all = {t["id"]: t for t in (tbls_link + tbls_sem)}

    top_img_ids = _cosine_top(q_vec, imgs_all, top_n=1)
    top_tbl_ids = _cosine_top(q_vec, tbls_all, top_n=1)

    img_item = imgs_all[top_img_ids[0]] if top_img_ids else None
    tbl_item = tbls_all[top_tbl_ids[0]] if top_tbl_ids else None

    # ---------------- prompt ----------------
    ctx: List[str] = []
    for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
        ctx.append(
            f"\n### Doc {i}\n"
            f"File: {meta.get('file_name', '')}\n"
            f"Chunk ID: {meta.get('chunk_id', '')[:8]}\n"
            f"---\n{doc[:1500]}\n"
        )

    if img_item:
        ctx.append(
            f"\n### Relevant Image\n"
            f"File: {img_item.get('file_name', '')}\n"
            f"Description: {img_item.get('summary', '')}\n"
        )

    if tbl_item:
        ctx.append(
            f"\n### Relevant Table\n"
            f"File: {tbl_item.get('file_name', '')}\n"
            f"Description: {tbl_item.get('summary', '')}\n"
        )

    prompt = f"""
You are a QA assistant for retrieved PDF content.

IMPORTANT RULES:
- Use only the provided material.
- Cite sources as (Doc 1), (Doc 2), etc.
- Do not invent facts, image contents, table values, URLs, or filenames beyond the material.
- If the answer is missing, say exactly:
  "Sorry, The text does not contain information about your question"

MATERIAL:
{''.join(ctx)}

Question:
{question}
"""

    try:
        answer = _chat(prompt).strip()
    except Exception:
        answer = ""

    if not answer:
        answer = "Sorry, The text does not contain information about your question"

    # ---------------- return media directly from retrieval ----------------
    show: List[Tuple[str, str]] = []

    if img_item:
        p = Path(img_item["path"])
        if p.exists():
            show.append(("img", str(p)))

    if tbl_item:
        p = Path(tbl_item["path"])
        if p.exists():
            show.append(("tbl", str(p)))

    return (answer, show) if return_media else answer




from __future__ import annotations

import base64
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import streamlit as st
from dotenv import load_dotenv

# ============================================================
# PATH SETUP
# ============================================================
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.rag_core import ingest_documents, smart_query  # noqa: E402

load_dotenv()

APP_TITLE = os.getenv("APP_TITLE", "PWC AI Platform")
DB_PATH = PROJECT_ROOT / "users.db"
UPLOAD_DIR = PROJECT_ROOT / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

AGENT_SCRIPT = PROJECT_ROOT / "agent.py"

st.set_page_config(page_title=APP_TITLE, layout="wide")


# ============================================================
# LOGO / HEADER
# ============================================================
def _find_logo() -> Path | None:
    candidates = []

    env_logo = os.getenv("APP_LOGO_PATH", "").strip()
    if env_logo:
        candidates.append(Path(env_logo))

    candidates.extend([
        APP_DIR / "assets" / "logo.png",
        PROJECT_ROOT / "assets" / "logo.png",
        APP_DIR / "logo.png",
        PROJECT_ROOT / "logo.png",
    ])

    for p in candidates:
        if p.exists():
            return p.resolve()
    return None


def _img_to_base64(img_path: Path) -> str:
    return base64.b64encode(img_path.read_bytes()).decode()


def render_header(app_mode: str):
    logo = _find_logo()
    if logo:
        b64 = _img_to_base64(logo)
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:16px;margin-bottom:8px;">
                <img src="data:image/png;base64,{b64}" width="80" style="border-radius:4px;">
                <div>
                    <div style="font-size:30px;font-weight:700;">{APP_TITLE}</div>
                    <div style="font-size:14px;color:#777;">{app_mode}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.title(APP_TITLE)
        st.caption(app_mode)


# ============================================================
# DATABASE
# ============================================================
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row

    # users
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            pw_hash TEXT
        )
    """)

    # conversations
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            mode TEXT NOT NULL,
            title TEXT DEFAULT 'New chat',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # chats
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER,
            role TEXT,
            payload TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # migration for older chats table
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(chats)").fetchall()}
    if "conversation_id" not in cols:
        try:
            conn.execute("ALTER TABLE chats ADD COLUMN conversation_id INTEGER")
        except Exception:
            pass
    if "created_at" not in cols:
        try:
            conn.execute("ALTER TABLE chats ADD COLUMN created_at TEXT DEFAULT CURRENT_TIMESTAMP")
        except Exception:
            pass

    conn.commit()
    return conn


def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def register_user(username: str, password: str) -> tuple[bool, str]:
    username = username.strip()
    if not username or not password:
        return False, "Username and password are required."

    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO users (username, pw_hash) VALUES (?, ?)",
            (username, hash_pw(password))
        )
        conn.commit()
        return True, "Registered successfully."
    except sqlite3.IntegrityError:
        return False, "Username already exists."
    finally:
        conn.close()


def login_user(username: str, password: str) -> tuple[bool, int | None, str]:
    username = username.strip()
    conn = get_db()
    row = conn.execute(
        "SELECT id, pw_hash FROM users WHERE username=?",
        (username,)
    ).fetchone()
    conn.close()

    if not row or row["pw_hash"] != hash_pw(password):
        return False, None, "Bad credentials."

    return True, int(row["id"]), "Login successful."


# ============================================================
# CONVERSATIONS / CHAT STORAGE
# ============================================================
def create_conversation(user_id: int, mode: str, title: str = "New chat") -> int:
    conn = get_db()
    cur = conn.execute(
        "INSERT INTO conversations (user_id, mode, title) VALUES (?, ?, ?)",
        (user_id, mode, title)
    )
    conn.commit()
    cid = int(cur.lastrowid)
    conn.close()
    return cid


def list_conversations(user_id: int, mode: str) -> list[tuple[int, str, str]]:
    conn = get_db()
    rows = conn.execute(
        """
        SELECT id, title, updated_at
        FROM conversations
        WHERE user_id=? AND mode=?
        ORDER BY updated_at DESC, id DESC
        """,
        (user_id, mode)
    ).fetchall()
    conn.close()
    return [(int(r["id"]), r["title"], r["updated_at"]) for r in rows]


def get_conversation_title(conversation_id: int) -> str:
    conn = get_db()
    row = conn.execute(
        "SELECT title FROM conversations WHERE id=?",
        (conversation_id,)
    ).fetchone()
    conn.close()
    return row["title"] if row else "New chat"


def set_conversation_title_if_new(conversation_id: int, text: str):
    current = get_conversation_title(conversation_id)
    if current and current != "New chat":
        return

    title = " ".join(text.strip().split())
    if not title:
        return
    title = title[:60]

    conn = get_db()
    conn.execute(
        "UPDATE conversations SET title=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
        (title, conversation_id)
    )
    conn.commit()
    conn.close()


def touch_conversation(conversation_id: int):
    conn = get_db()
    conn.execute(
        "UPDATE conversations SET updated_at=CURRENT_TIMESTAMP WHERE id=?",
        (conversation_id,)
    )
    conn.commit()
    conn.close()


def save_chat(conversation_id: int, role: str, payload):
    if not isinstance(payload, str):
        payload = json.dumps(payload, ensure_ascii=False)

    conn = get_db()
    conn.execute(
        "INSERT INTO chats (conversation_id, role, payload) VALUES (?, ?, ?)",
        (conversation_id, role, payload)
    )
    conn.execute(
        "UPDATE conversations SET updated_at=CURRENT_TIMESTAMP WHERE id=?",
        (conversation_id,)
    )
    conn.commit()
    conn.close()


def load_chats(conversation_id: int) -> list[tuple[str, str]]:
    conn = get_db()
    rows = conn.execute(
        """
        SELECT role, payload
        FROM chats
        WHERE conversation_id=?
        ORDER BY id
        """,
        (conversation_id,)
    ).fetchall()
    conn.close()
    return [(r["role"], r["payload"]) for r in rows]


def delete_conversation(conversation_id: int):
    conn = get_db()
    conn.execute("DELETE FROM chats WHERE conversation_id=?", (conversation_id,))
    conn.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))
    conn.commit()
    conn.close()


def ensure_active_conversation(user_id: int, mode: str) -> int:
    current = st.session_state.get("deep_conversation_id")

    if current is not None:
        conn = get_db()
        row = conn.execute(
            "SELECT id FROM conversations WHERE id=? AND user_id=? AND mode=?",
            (current, user_id, mode)
        ).fetchone()
        conn.close()
        if row:
            return current

    convs = list_conversations(user_id, mode)
    if convs:
        st.session_state.deep_conversation_id = convs[0][0]
        return convs[0][0]

    cid = create_conversation(user_id, mode)
    st.session_state.deep_conversation_id = cid
    return cid


# ============================================================
# SESSION STATE
# ============================================================
if "agent_history" not in st.session_state:
    st.session_state.agent_history = []

if "uid" not in st.session_state:
    st.session_state.uid = None

if "username" not in st.session_state:
    st.session_state.username = None

if "deep_conversation_id" not in st.session_state:
    st.session_state.deep_conversation_id = None

if "flash_message" not in st.session_state:
    st.session_state.flash_message = ""


# ============================================================
# RENDER HELPERS
# ============================================================
MEDIA_TOKEN_RE = st.session_state.get(
    "_media_token_re",
    None
)
if MEDIA_TOKEN_RE is None:
    import re
    MEDIA_TOKEN_RE = re.compile(r"^\s*<<(img|tbl):[^>]+>>\s*$", re.MULTILINE)
    st.session_state._media_token_re = MEDIA_TOKEN_RE


def clean_answer_text(text: str) -> str:
    return MEDIA_TOKEN_RE.sub("", text).strip()


def render_media(media: List[Tuple[str, str]] | List[list]):
    for item in media:
        kind, path = item
        p = Path(path)
        if not p.exists():
            continue

        if kind == "img":
            st.image(str(p), use_container_width=True)
        elif kind == "tbl":
            with st.expander(f"Table: {p.name}", expanded=False):
                st.markdown(p.read_text(encoding="utf-8"))


def render_saved_message(role: str, payload: str):
    if role == "user":
        st.markdown(payload)
        return

    try:
        data = json.loads(payload)
        answer = data.get("answer", "")
        media = data.get("media", [])
        st.markdown(answer)
        render_media(media)
    except Exception:
        st.markdown(payload)


# ============================================================
# AUTH UI
# ============================================================
def common_auth():
    if st.session_state.uid is not None:
        return

    _, mid, _ = st.columns([1, 2, 1])

    with mid:
        logo = _find_logo()
        if logo:
            c1, c2 = st.columns([0.18, 0.82])
            with c1:
                st.image(str(logo), width=90)
            with c2:
                st.markdown(
                    f"<h2 style='margin-top:18px;'>{APP_TITLE} Login</h2>",
                    unsafe_allow_html=True,
                )
        else:
            st.subheader(f"{APP_TITLE} Login")

        tab1, tab2 = st.tabs(["Login", "Register"])

        with tab1:
            u = st.text_input("Username", key="login_user")
            p = st.text_input("Password", type="password", key="login_pw")
            if st.button("Login", key="login_btn", use_container_width=True):
                ok, uid, msg = login_user(u, p)
                if ok:
                    st.session_state.uid = uid
                    st.session_state.username = u.strip()
                    st.session_state.deep_conversation_id = None
                    st.rerun()
                else:
                    st.error(msg)

        with tab2:
            u = st.text_input("New Username", key="reg_user")
            p = st.text_input("New Password", type="password", key="reg_pw")
            if st.button("Register", key="reg_btn", use_container_width=True):
                ok, msg = register_user(u, p)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

    st.stop()


# ============================================================
# AGENTIC MODE
# ============================================================
def run_agentic_mode():
    for msg in st.session_state.agent_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Enter instructions for Agentic MDAO...", key="agent_input")
    if not prompt:
        return

    st.session_state.agent_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_output = ""

        try:
            process = subprocess.Popen(
                [sys.executable, str(AGENT_SCRIPT), prompt],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            for line in iter(process.stdout.readline, ""):
                if not line:
                    break
                full_output += line
                response_placeholder.markdown(full_output + "▌")

            process.wait()
            full_output = full_output.strip() or "No response returned."
            response_placeholder.markdown(full_output)

        except Exception as e:
            full_output = f"Error running agent script: {e}"
            response_placeholder.markdown(full_output)

    st.session_state.agent_history.append({"role": "assistant", "content": full_output})


# ============================================================
# DEEP RESEARCH SIDEBAR
# ============================================================
def render_deep_sidebar(uid: int, mode: str):
    active_id = ensure_active_conversation(uid, mode)

    with st.sidebar:
        st.divider()
        st.markdown("### Chats")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("New Chat", use_container_width=True, key="new_deep_chat"):
                st.session_state.deep_conversation_id = create_conversation(uid, mode)
                st.rerun()

        with c2:
            if st.button("Delete Chat", use_container_width=True, key="delete_deep_chat"):
                delete_conversation(active_id)
                st.session_state.deep_conversation_id = None
                ensure_active_conversation(uid, mode)
                st.rerun()

        convs = list_conversations(uid, mode)
        if not convs:
            st.caption("No chats yet.")
            return

        for conv_id, title, updated_at in convs:
            label = f"▶ {title}" if conv_id == active_id else title
            if st.button(label, key=f"conv_{conv_id}", use_container_width=True):
                st.session_state.deep_conversation_id = conv_id
                st.rerun()


# ============================================================
# DEEP RESEARCH MODE
# ============================================================
def run_deep_research_mode():
    uid = st.session_state.uid
    mode = "deep_research"

    render_deep_sidebar(uid, mode)
    conversation_id = ensure_active_conversation(uid, mode)
    user_dir = UPLOAD_DIR / f"user_{uid}"
    user_dir.mkdir(parents=True, exist_ok=True)

    current_title = get_conversation_title(conversation_id)
    st.subheader(current_title)

    if st.session_state.flash_message:
        st.success(st.session_state.flash_message)
        st.session_state.flash_message = ""

    # upload moved to main area, not sidebar
    with st.expander("Upload and index PDF", expanded=False):
        st.caption("Uploaded PDFs stay private to your account.")
        uploaded_pdf = st.file_uploader(
            "Choose a PDF",
            type=["pdf"],
            key="deep_pdf_upload"
        )

        if uploaded_pdf and st.button("Save + Ingest", use_container_width=True, key="deep_ingest_btn"):
            save_path = user_dir / uploaded_pdf.name
            save_path.write_bytes(uploaded_pdf.getbuffer())

            with st.spinner(f"Ingesting {uploaded_pdf.name}..."):
                ingest_documents(str(save_path), user_id=uid)

            st.session_state.flash_message = f"Indexed: {uploaded_pdf.name}"
            st.rerun()

    # replay saved history
    history = load_chats(conversation_id)
    if not history:
        st.info("Start a new chat by asking about your indexed PDFs.")

    for role, payload in history:
        with st.chat_message("assistant" if role == "assistant" else "user"):
            render_saved_message(role, payload)

    # ask question
    prompt = st.chat_input("Ask about your uploaded PDFs...", key="deep_chat_input")
    if not prompt:
        return

    set_conversation_title_if_new(conversation_id, prompt)
    save_chat(conversation_id, "user", prompt)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                answer, media = smart_query(prompt, user_id=uid, return_media=True)

            clean_answer = clean_answer_text(answer)
            st.markdown(clean_answer)
            render_media(media)

            save_chat(
                conversation_id,
                "assistant",
                {"answer": clean_answer, "media": media}
            )

        except Exception as e:
            err = f"Server error: {e}"
            st.error(err)
            save_chat(
                conversation_id,
                "assistant",
                {"answer": err, "media": []}
            )


# ============================================================
# MAIN
# ============================================================
common_auth()

with st.sidebar:
    st.write(f"Welcome, **{st.session_state.username}**")
    app_mode = st.radio(
        "Choose Mode",
        ["Agentic MDAO", "Deep Research"],
        key="main_mode_radio"
    )
    if st.button("Logout", use_container_width=True, key="logout_btn"):
        st.session_state.uid = None
        st.session_state.username = None
        st.session_state.deep_conversation_id = None
        st.session_state.agent_history = []
        st.rerun()

render_header(app_mode)

if app_mode == "Agentic MDAO":
    run_agentic_mode()
else:
    run_deep_research_mode()