from pathlib import Path
from typing import Iterable
from app.eca.lightrag_bootstrap import get_rag
from concurrent.futures import ThreadPoolExecutor
import uuid
# minimal text extraction helpers (keep it lightweight/offline)
def read_text(path: Path) -> str:
   suffix = path.suffix.lower()
   if suffix in {".txt", ".md"}:
       return path.read_text(errors="ignore")
   elif suffix == ".pdf":
       try:
           from pypdf import PdfReader
           txt = []
           r = PdfReader(str(path))
           for page in r.pages:
               txt.append(page.extract_text() or "")
           return "\n".join(txt)
       except Exception:
           return ""  # skip unreadable
   elif suffix in {".docx"}:
       try:
           import docx
           d = docx.Document(str(path))
           return "\n".join(p.text for p in d.paragraphs)
       except Exception:
           return ""
   elif suffix in {".csv"}:
       try:
           return path.read_text(errors="ignore")
       except Exception:
           return ""
   else:
       return ""  # unknown formats ignored (keep simple)
def iter_docs(folder: Path) -> Iterable[tuple[str, str]]:
   for p in folder.rglob("*"):
       if not p.is_file():
           continue
       t = read_text(p)
       if t.strip():
           yield (str(p), t)
def index_folder(folder_path: str, parallel: int = 4) -> dict:
   folder = Path(folder_path).expanduser().resolve()
   assert folder.exists(), f"Folder not found: {folder}"
   rag = get_rag()
   # LightRAG supports `insert()` with text; we tag with doc_id=file path
   items = list(iter_docs(folder))
   if not items:
       return {"indexed": 0, "skipped": 0}
   def _insert_one(item):
       file_path, text = item
       try:
           # doc_id helps with later deletion or updates
           rag.insert(text, doc_id=file_path)
           return True
       except Exception:
           return False
   ok = 0
   with ThreadPoolExecutor(max_workers=parallel) as ex:
       for done in ex.map(_insert_one, items):
           ok += 1 if done else 0
   return {"indexed": ok, "skipped": len(items) - ok}