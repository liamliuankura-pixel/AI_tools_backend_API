from typing import List
from app.eca.config import cfg
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
import ollama  # python SDK
def _chat_complete(prompt: str) -> str:
   """
   LightRAG expects a callable(prompt) -> str.
   We use ollama.chat() via the Python SDK, no manual HTTP.
   """
   # (optional) you can add a system message for stable behavior
   messages = [
       {"role": "system", "content": "You are a precise legal-technology assistant for Early Case Assessment."},
       {"role": "user", "content": prompt},
   ]
   resp = ollama.chat(
       model=cfg.LLM_MODEL,
       messages=messages,
       options={"num_ctx": cfg.OLLAMA_NUM_CTX},
   )
   return resp["message"]["content"]
def _embed_texts(texts: List[str]) -> List[List[float]]:
   """
   Batch-embed by calling ollama.embeddings per text (SDK is synchronous).
   """
   vecs: List[List[float]] = []
   for t in texts:
       e = ollama.embeddings(model=cfg.EMBED_MODEL, prompt=t)
       vecs.append(e["embedding"])
   return vecs
# single LightRAG instance
_rag = None
def build_rag() -> LightRAG:
   global _rag
   if _rag is not None:
       return _rag
   _rag = LightRAG(
       working_dir=str(cfg.WORKING_DIR),
       llm_model_func=_chat_complete,
       embedding_func=EmbeddingFunc(
           embedding_dim=cfg.EMBED_DIM,
           max_token_size=8192,
           func=_embed_texts,
       ),
   )
   return _rag
def get_rag() -> LightRAG:
   return build_rag()