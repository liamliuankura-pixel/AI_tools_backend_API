from typing import List, Literal, Callable, Dict
from app.eca.config import cfg
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
import ollama
# --- Model router -------------------------------------------------------------
ModelRole = Literal["base", "sum", "expert"]
print(cfg.BASE_LLM_MODEL, cfg.SUM_LLM_MODEL, cfg.EXPERT_LLM_MODEL)
MODEL_MAP: Dict[ModelRole, str] = {
   "base":   cfg.BASE_LLM_MODEL,
   "sum":    cfg.SUM_LLM_MODEL,
   "expert": cfg.EXPERT_LLM_MODEL,
}
def _chat_complete_with_role(prompt: str, role: ModelRole = "base") -> str:
   model_name = MODEL_MAP.get(role, cfg.BASE_LLM_MODEL)
   messages = [
       {"role": "system", "content": (
           "You are an accurate, concise assistant for Early Case Assessment. "
           "Prefer faithful summaries and cite entities and relationships when possible."
       )},
       {"role": "user", "content": prompt},
   ]
   resp = ollama.chat(
       model=model_name,
       messages=messages,
       options={"num_ctx": cfg.OLLAMA_NUM_CTX},
   )
   return resp["message"]["content"]
# Adapter: LightRAG expects a no-arg signature except prompt.
# We capture a default 'role' but allow callers to override via closure.
def make_llm(role: ModelRole) -> Callable[[str], str]:
   def _call(prompt: str) -> str:
       return _chat_complete_with_role(prompt, role=role)
   return _call
def _embed_texts(texts: List[str]) -> List[List[float]]:
   vecs: List[List[float]] = []
   for t in texts:
       e = ollama.embeddings(model=cfg.EMBED_MODEL, prompt=t)
       vecs.append(e["embedding"])
   return vecs
# --- LightRAG singleton -------------------------------------------------------
_rag: LightRAG | None = None
def build_rag() -> LightRAG:
   global _rag
   if _rag is not None:
       return _rag
   # Default LLM for LightRAG internal prompts -> use "base"
   _rag = LightRAG(
       working_dir=str(cfg.WORKING_DIR),
       llm_model_func=make_llm("base"),
       embedding_func=EmbeddingFunc(
           embedding_dim=cfg.EMBED_DIM,
           max_token_size=8192,
           func=_embed_texts,
       ),
   )
   return _rag
def get_rag() -> LightRAG:
   return build_rag()