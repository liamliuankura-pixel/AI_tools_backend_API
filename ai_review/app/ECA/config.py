from pathlib import Path
from pydantic import BaseModel
import os
class ECAConfig(BaseModel):
   # storage for LightRAG (SQLite DB, index, graph)
   WORKING_DIR: Path = Path(os.getenv("ECA_WORKING_DIR", ".eca_store")).resolve()
   # three-model routing (all local via Ollama)
   BASE_LLM_MODEL: str   = os.getenv("ECA_BASE_LLM_MODEL",   "qwen2.5:7b-instruct")
   SUM_LLM_MODEL: str    = os.getenv("ECA_SUM_LLM_MODEL",    "qwen2.5:7b-instruct")
   EXPERT_LLM_MODEL: str = os.getenv("ECA_EXPERT_LLM_MODEL", "qwen2.5:7b-instruct")
   # embeddings via ollama
   EMBED_MODEL: str = os.getenv("ECA_EMBED_MODEL", "nomic-embed-text")
   EMBED_DIM: int   = int(os.getenv("ECA_EMBED_DIM", "768"))  # nomic-embed-text = 768
   # retrieval strategy: naive|local|global (LightRAG modes)
   RETRIEVAL_MODE: str = os.getenv("ECA_RETRIEVAL_MODE", "global")
   # ollama context window (API options)
   OLLAMA_NUM_CTX: int = int(os.getenv("ECA_OLLAMA_NUM_CTX", "32768"))
   # ingestion parallelism
   PARALLEL: int = int(os.getenv("ECA_PARALLEL", "4"))
cfg = ECAConfig()
cfg.WORKING_DIR.mkdir(parents=True, exist_ok=True)