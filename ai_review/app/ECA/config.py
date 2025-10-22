from pathlib import Path
from pydantic import BaseModel
import os
class ECAConfig(BaseModel):
   WORKING_DIR: Path = Path(os.getenv("ECA_WORKING_DIR", ".eca_store")).resolve()
   # python SDK talks to local daemon; leave for completeness / remote sockets
   OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
   LLM_MODEL: str = os.getenv("ECA_LLM_MODEL", "qwen2.5:7b-instruct")
   EMBED_MODEL: str = os.getenv("ECA_EMBED_MODEL", "nomic-embed-text")
   EMBED_DIM: int = int(os.getenv("ECA_EMBED_DIM", "768"))  # nomic-embed-text = 768
   RETRIEVAL_MODE: str = os.getenv("ECA_RETRIEVAL_MODE", "global")
   OLLAMA_NUM_CTX: int = int(os.getenv("ECA_OLLAMA_NUM_CTX", "32768"))
   PARALLEL: int = int(os.getenv("ECA_PARALLEL", "4"))
cfg = ECAConfig()
cfg.WORKING_DIR.mkdir(parents=True, exist_ok=True)