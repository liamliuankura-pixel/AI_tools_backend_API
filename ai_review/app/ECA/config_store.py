# app/ECA/config_store.py

from __future__ import annotations

import hashlib

import json

import re

import sqlite3

from datetime import datetime

from pathlib import Path

from typing import Dict, List

from fastapi import HTTPException

from pydantic import BaseModel

# ============================== CONFIG ==============================

class ECAConfig(BaseModel):

    case_id: str

    data_root: str = "data"

    chunk_chars: int = 1200

    chunk_overlap: int = 200

    embedding_backend: str = "ollama_py"  # "ollama_py" | "sbert"

    embedding_model: str = "nomic-embed-text"

    sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    use_faiss: bool = True

    enable_bm25: bool = True

    enable_graph_spread: bool = True


# ============================== STORAGE =============================

class Store:

    """SQLite for docs/chunks/settings; optional FAISS file.

    Files live in: {data_root}/{case_id}/

    """

    def __init__(self, cfg: ECAConfig):

        self.cfg = cfg

        self.case_dir = Path(cfg.data_root) / cfg.case_id

        self.case_dir.mkdir(parents=True, exist_ok=True)

        self.db = self.case_dir / "index.sqlite"

        self.faiss_path = self.case_dir / "vectors.faiss"

        self.bm25_path = self.case_dir / "bm25.jsonl"

        self.graph_path = self.case_dir / "graph.jsonl"

        self.conn = sqlite3.connect(self.db)

        self._init_db()

    def _init_db(self):

        c = self.conn.cursor()

        c.execute("""

        CREATE TABLE IF NOT EXISTS docs(

            doc_id TEXT PRIMARY KEY,

            path   TEXT,

            sha1   TEXT,

            size   INTEGER,

            mtime  REAL,

            meta   JSON

        )""")

        c.execute("""

        CREATE TABLE IF NOT EXISTS chunks(

            chunk_id TEXT PRIMARY KEY,

            doc_id   TEXT,

            ord      INTEGER,

            text     TEXT,

            meta     JSON

        )""")

        c.execute("""CREATE TABLE IF NOT EXISTS settings(k TEXT PRIMARY KEY, v TEXT)""")

        self.conn.commit()

    # ---- metadata

    def put_meta(self, k: str, v: Dict):

        c = self.conn.cursor()

        c.execute("REPLACE INTO settings(k,v) VALUES(?,?)", (k, json.dumps(v)))

        self.conn.commit()

    def get_meta(self, k: str, default=None):

        c = self.conn.cursor()

        c.execute("SELECT v FROM settings WHERE k=?", (k,))

        row = c.fetchone()

        return json.loads(row[0]) if row else default

    # ---- docs

    def upsert_doc(self, doc_id: str, path: str, sha1: str, size: int, mtime: float, meta=None):

        c = self.conn.cursor()

        c.execute("""REPLACE INTO docs(doc_id,path,sha1,size,mtime,meta)

                     VALUES(?,?,?,?,?,?)""",

                  (doc_id, path, sha1, size, mtime, json.dumps(meta or {})))

        self.conn.commit()

    def list_docs(self) -> List[Dict]:

        c = self.conn.cursor()

        c.execute("SELECT doc_id,path,sha1,size,mtime,meta FROM docs")

        return [dict(doc_id=r[0], path=r[1], sha1=r[2], size=r[3], mtime=r[4], meta=json.loads(r[5] or "{}"))

                for r in c.fetchall()]

    # ---- chunks

    def upsert_chunk(self, chunk_id: str, doc_id: str, ord_: int, text: str, meta=None):

        c = self.conn.cursor()

        c.execute("""REPLACE INTO chunks(chunk_id,doc_id,ord,text,meta)

                     VALUES(?,?,?,?,?)""",

                  (chunk_id, doc_id, ord_, text, json.dumps(meta or {})))

        self.conn.commit()

    def list_chunks(self) -> List[Dict]:

        c = self.conn.cursor()

        c.execute("SELECT chunk_id,doc_id,ord,text,meta FROM chunks")

        return [dict(chunk_id=r[0], doc_id=r[1], ord=r[2], text=r[3], meta=json.loads(r[4] or "{}"))

                for r in c.fetchall()]


# ============================== LIGHT EXTRACT ========================

try:

    import pypdf  # optional

except Exception:

    pypdf = None

def _sha1(path: Path) -> str:

    h = hashlib.sha1()

    with path.open("rb") as f:

        for b in iter(lambda: f.read(1 << 20), b""):

            h.update(b)

    return h.hexdigest()

def _clean(t: str) -> str:

    t = t.replace("\x00", " ")

    t = re.sub(r"[ \t]+", " ", t)

    t = re.sub(r"\n{3,}", "\n\n", t)

    return t.strip()

def _chunk(text: str, size=1200, overlap=200):

    if len(text) <= size:

        return [text]

    out, i = [], 0

    while i < len(text):

        j = min(len(text), i + size)

        out.append(text[i:j])

        if j == len(text):

            break

        i = max(0, j - overlap)

    return out

def _read_txt(p: Path) -> str:

    return p.read_text(errors="ignore")

def _read_pdf(p: Path) -> str:

    if not pypdf:

        raise RuntimeError("pypdf not installed")

    reader = pypdf.PdfReader(str(p))

    return "\n\n".join((pg.extract_text() or "") for pg in reader.pages)

READERS = {

    ".txt": _read_txt,

    ".md":  _read_txt,

    ".log": _read_txt,

    ".csv": _read_txt,

    ".pdf": _read_pdf,

}

# ============================== TEMP ECA SCAN ========================

def scan_folder_to_chunks(cfg: ECAConfig, store: Store, folder: str) -> Dict:

    """

    Temporary local scan for ECA. Later, replace this with Relativity Data API

    ingestion that writes into the same docs/chunks tables.

    """

    root = Path(folder)

    if not root.is_dir():

        raise HTTPException(400, f"Folder not found: {folder}")

    ndocs, nchunks = 0, 0

    for path in root.rglob("*"):

        if not path.is_file():

            continue

        ext = path.suffix.lower()

        if ext not in READERS:

            continue  # skip binaries for now

        size, mtime = path.stat().st_size, path.stat().st_mtime

        sha1 = _sha1(path)

        doc_id = sha1[:12]

        try:

            text = READERS[ext](path)

            text = _clean(text)

        except Exception as e:

            store.upsert_doc(doc_id, str(path), sha1, size, mtime, meta={"error": str(e)})

            continue

        store.upsert_doc(doc_id, str(path), sha1, size, mtime, meta={"ext": ext})

        for i, chunk in enumerate(_chunk(text, cfg.chunk_chars, cfg.chunk_overlap)):

            store.upsert_chunk(f"{doc_id}:{i:05d}", doc_id, i, chunk, meta={"source_path": str(path)})

            nchunks += 1

        ndocs += 1

    return {"docs": ndocs, "chunks": nchunks}
 