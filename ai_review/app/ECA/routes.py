from fastapi import APIRouter, HTTPException
from app.eca.schemas import IndexRequest, QueryRequest
from app.eca.config import cfg
from app.eca.ingest import index_folder
from app.eca.lightrag_bootstrap import get_rag, make_llm
router = APIRouter(prefix="/eca", tags=["eca"])
@router.get("/health")
def health():
   return {
       "status": "ok",
       "working_dir": str(cfg.WORKING_DIR),
       "embed_model": cfg.EMBED_MODEL,
       "embed_dim": cfg.EMBED_DIM,
       "default_retrieval_mode": cfg.RETRIEVAL_MODE,
       "models": {
           "base": cfg.BASE_LLM_MODEL,
           "sum": cfg.SUM_LLM_MODEL,
           "expert": cfg.EXPERT_LLM_MODEL,
       },
       "num_ctx": cfg.OLLAMA_NUM_CTX
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
   # temporarily override the LLM used for this query
   rag.llm_model_func = make_llm(req.model_role or "base")
   if mode == "naive":
       out = rag.query(req.question, retrieval_strategy="naive", top_k=req.top_k)
   elif mode == "local":
       out = rag.query(req.question, retrieval_strategy="local", top_k=req.top_k)
   else:
       out = rag.query(req.question, retrieval_strategy="global", top_k=req.top_k)
   return {"mode": mode, "model_role": req.model_role or "base", "answer": str(out)}