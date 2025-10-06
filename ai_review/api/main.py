from fastapi import FastAPI
import os
import logging
from logging.handlers import RotatingFileHandler
# Build a path like ai_review/logs/app.log regardless of where the code runs
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # -> ai_review
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "app.log")
# Rotating log: 5 MB per file, keep 5 backups
handler = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
logging.basicConfig(
   level=logging.INFO,
   format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
   handlers=[handler]  # only to file; add logging.StreamHandler() if you still want console
)
logger = logging.getLogger(__name__)

from typing import List, Optional
from app.schemas import ReviewRequest, ReviewResponse, DocResult
from app.models import EnsembleReviewer
from app.providers.ollama_client import OllamaClient
from pydantic import BaseModel
from typing import Dict, Any

from app.ECA.routes import router as eca_router

app = FastAPI(title="Responsiveness & ECA API", version="0.3.0")
app.include_router(eca_router)
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
            logger.info("Review output for %s: %s", d.filename, out)
            results.append(DocResult(filename=d.filename, final_label=out["final"],
                                     votes=out["votes"], rationales=out["rationales"]))
    elif mode == "collab":
        names = req.model_names or MODEL_DEFAULTS
        arbiter = req.model_name or names[0]
        for d in req.docs:
            out = engine.review_doc_collab(names, req.case_background, d.text,
                                           arbiter=arbiter, critique_rounds=1)
            logger.info("Review output for %s: %s", d.filename, out)
            rats = {**out["rationales"], "__arbiter__": out.get("arbiter_rationale", "")}
            results.append(DocResult(filename=d.filename, final_label=out["final"],
                                     votes=out["votes"], rationales=rats))
    else:  # ensemble (majority)
        names = req.model_names or MODEL_DEFAULTS
        for d in req.docs:
            out = engine.review_doc_ensemble(names, req.case_background, d.text)
            logger.info("Review output for %s: %s", d.filename, out)
            results.append(DocResult(filename=d.filename, final_label=out["final"],
                                     votes=out["votes"], rationales=out["rationales"]))
    
    return ReviewResponse(results=results)


