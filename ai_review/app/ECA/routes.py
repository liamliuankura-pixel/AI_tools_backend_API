# app/ECA/routes.py
from __future__ import annotations
from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field
from .config_store import ECAConfig, Store, scan_folder_to_chunks
from .pipeline import (
   EmbeddingBackend,
   build_bm25,
   build_graph,
   build_vector_index,
   hybrid_search,
)
router = APIRouter(prefix="/eca", tags=["ECA"])

# ============================== SCHEMAS ==============================
class InitReq(BaseModel):
   case_id: str
   background: str = Field("", description="Case background / prior info")
   config_overrides: Optional[dict] = None
class ScanReq(BaseModel):
   case_id: str
   folder: str  # temporary: local scan; replace with RDA later
class BuildReq(BaseModel):
   case_id: str
class SearchReq(BaseModel):
   case_id: str
   query: str
   top_k: int = 20
   graph_weight: float = 0.6  # weight of PPR signal in fusion

# ============================== HELPERS ==============================
def _cfg_store(case_id: str, overrides: Optional[dict] = None):
   cfg = ECAConfig(case_id=case_id, **(overrides or {}))
   return cfg, Store(cfg)

# ============================== ENDPOINTS ==============================
@router.post("/init")
def init_case(req: InitReq):
   cfg, store = _cfg_store(req.case_id, req.config_overrides)
   store.put_meta("case_background", {"text": req.background})
   return {"ok": True, "case_id": req.case_id, "dir": str(store.case_dir)}
@router.post("/scan")
def scan(req: ScanReq):
   cfg, store = _cfg_store(req.case_id)
   res = scan_folder_to_chunks(cfg, store, req.folder)
   return {"ok": True, **res}
@router.post("/build")
def build(req: BuildReq):
   cfg, store = _cfg_store(req.case_id)
   eb = EmbeddingBackend(cfg)
   v = build_vector_index(cfg, store, eb)
   b = build_bm25(cfg, store)
   return {"ok": True, "vectors": v, "bm25": b}
@router.post("/graph")
def build_graph_api(req: BuildReq):
   cfg, store = _cfg_store(req.case_id)
   g = build_graph(store)
   return {"ok": True, **g}
@router.post("/search")
def search(req: SearchReq):
   cfg, store = _cfg_store(req.case_id)
   eb = EmbeddingBackend(cfg)
   bg = store.get_meta("case_background", {}).get("text", "")
   hits = hybrid_search(
       cfg, store, eb,
       req.query, bg,
       top_k=req.top_k,
       graph_weight=req.graph_weight
   )
   return {"ok": True, "background_used": bool(bg), "hits": hits}