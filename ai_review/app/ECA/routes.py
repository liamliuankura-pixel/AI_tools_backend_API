from fastapi import APIRouter, HTTPException
from app.eca.schemas import IndexRequest, QueryRequest
from app.eca.config import cfg
from app.eca.ingest import index_folder
from app.eca.lightrag_bootstrap import get_rag
router = APIRouter(prefix="/eca", tags=["eca"])
@router.get("/health")
def health():
   return {
       "status": "ok",
       "llm_model": cfg.LLM_MODEL,
       "embed_model": cfg.EMBED_MODEL,
       "working_dir": str(cfg.WORKING_DIR),
       "retrieval_mode": cfg.RETRIEVAL_MODE
   }
@router.post("/index")
def index_docs(req: IndexRequest):
   try:
       stats = index_folder(req.folder, parallel=cfg.PARALLEL)
       return {"ok": True, "stats": stats}
   except AssertionError as e:
       raise HTTPException(status_code=400, detail=str(e))
@router.post("/query")
def query(req: QueryRequest):
   rag = get_rag()
   mode = (req.mode or cfg.RETRIEVAL_MODE).lower()
   # LightRAG has several modes; keep the switch explicit
   if mode == "naive":
       answer = rag.query(req.question, retrieval_strategy="naive")
   elif mode == "local":
       answer = rag.query(req.question, retrieval_strategy="local")
   else:
       # default/global (graph-augmented)
       answer = rag.query(req.question, retrieval_strategy="global")
   # answer is usually a dict/string per LightRAG; normalize to str
   return {"mode": mode, "answer": str(answer)}