from fastapi import FastAPI
from typing import List, Optional
from app.schemas import ReviewRequest, ReviewResponse, DocResult
from app.models import EnsembleReviewer
from app.providers.ollama_client import OllamaClient
from pydantic import BaseModel
from typing import Dict, Any
from app.eca_rag.engine import ECARAG
app = FastAPI(title="Responsiveness & ECA API", version="0.3.0")
MODEL_DEFAULTS = ["gemma3n:e4b", "llama3.2:3b", "qwen3:4b"]
# ---------- Health & Config ----------
@app.get("/health")
def health():
    return {"ok": True, "version": "2025-09-16"}
@app.get("/models")
def models():
    return {"defaults": MODEL_DEFAULTS}
# ---------- Responsiveness Review ----------
@app.post("/review", response_model=ReviewResponse)
def review(req: ReviewRequest):
    client = OllamaClient()
    engine = EnsembleReviewer(client)
    results: List[DocResult] = []
    mode = (req.mode or "ensemble").lower()
    if mode == "single":
        model_name = req.model_name or MODEL_DEFAULTS[0]
        for d in req.docs:
            out = engine.review_doc_single(model_name, req.case_background, d.text)
            results.append(DocResult(filename=d.filename, final_label=out["final"],
                                     votes=out["votes"], rationales=out["rationales"]))
    elif mode == "collab":
        names = req.model_names or MODEL_DEFAULTS
        arbiter = req.model_name or names[0]
        for d in req.docs:
            out = engine.review_doc_collab(names, req.case_background, d.text,
                                           arbiter=arbiter, critique_rounds=1)
            rats = {**out["rationales"], "__arbiter__": out.get("arbiter_rationale", "")}
            results.append(DocResult(filename=d.filename, final_label=out["final"],
                                     votes=out["votes"], rationales=rats))
    else:  # ensemble (majority)
        names = req.model_names or MODEL_DEFAULTS
        for d in req.docs:
            out = engine.review_doc_ensemble(names, req.case_background, d.text)
            results.append(DocResult(filename=d.filename, final_label=out["final"],
                                     votes=out["votes"], rationales=out["rationales"]))
    return ReviewResponse(results=results)
# ---------- ECA / RAG ----------
# ECA = ECARAG(gen_model="gemma3n:e4b", embed_model="nomic-embed-text")
# class IngestDoc(BaseModel):
#     id: str
#     text: str
#     meta: Optional[Dict[str, Any]] = None
# class IngestRequest(BaseModel):
#     docs: List[IngestDoc]
# class QueryRequest(BaseModel):
#     query: str
#     k: int = 5
# class SummaryRequest(BaseModel):
#     case_background: str
#     query: str
#     k: int = 6
# @app.post("/eca/ingest")
# def eca_ingest(req: IngestRequest):
#     return ECA.ingest_docs([d.model_dump() for d in req.docs])
# @app.post("/eca/query")
# def eca_query(req: QueryRequest):
#     hits = ECA.retrieve(req.query, k=req.k)
#     return {"results": [{"id": h["id"], "score": h["score"], "meta": h["meta"]} for h in hits]}
# @app.post("/eca/summary")
# def eca_summary(req: SummaryRequest):
#     return ECA.summarize(req.case_background, req.query, k=req.k)
# from typing import List, Optional, Dict
# from pydantic import BaseModel, Field
# class DocInput(BaseModel):
#     filename: str
#     text: str
# class ReviewRequest(BaseModel):
#     mode: str = Field(default="ensemble", description="'single' | 'ensemble' | 'collab'")
#     case_background: str
#     docs: List[DocInput]
#     # for 'single'
#     model_name: Optional[str] = None
#     # for 'ensemble' / 'collab'
#     model_names: Optional[List[str]] = None
# class DocResult(BaseModel):
#     filename: str
#     final_label: str
#     votes: Dict[str, str]
#     rationales: Dict[str, str]
# class ReviewResponse(BaseModel):
#     results: List[DocResult]
 