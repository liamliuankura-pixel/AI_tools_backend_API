# app/ECA/pipeline.py
from __future__ import annotations
import json
import re
from collections import defaultdict
from math import sqrt
from typing import Dict, List, Tuple
from .config_store import ECAConfig, Store
# ---------- Embedding backends ----------
try:
   import ollama  # Python package (daemon must be running; model pulled)
except Exception:
   ollama = None
try:
   from sentence_transformers import SentenceTransformer
except Exception:
   SentenceTransformer = None
# ---------- Optional deps ----------
try:
   import faiss, numpy as np
except Exception:
   faiss, np = None, None
try:
   from rank_bm25 import BM25Okapi
except Exception:
   BM25Okapi = None
try:
   import spacy
   _NLP = spacy.load("en_core_web_sm")
except Exception:
   _NLP = None

# ============================== EMBEDDINGS ==============================
class EmbeddingBackend:
   def __init__(self, cfg: ECAConfig):
       self.cfg = cfg
       self._sbert = None
       if cfg.embedding_backend == "sbert":
           if not SentenceTransformer:
               raise RuntimeError("sentence-transformers not installed")
           self._sbert = SentenceTransformer(cfg.sbert_model)
       if cfg.embedding_backend == "ollama_py" and not ollama:
           raise RuntimeError("`ollama` Python package not installed")
   def embed(self, texts: List[str]) -> List[List[float]]:
       if self.cfg.embedding_backend == "ollama_py":
           return self._embed_ollama_py(texts)
       return self._embed_sbert(texts)
   def _embed_ollama_py(self, texts: List[str]) -> List[List[float]]:
       # Requires: `ollama serve` and `ollama pull nomic-embed-text`
       out = []
       for t in texts:
           r = ollama.embeddings(model=self.cfg.embedding_model, prompt=t)
           out.append(r["embedding"])
       return out
   def _embed_sbert(self, texts: List[str]) -> List[List[float]]:
       mat = self._sbert.encode(texts, show_progress_bar=False, normalize_embeddings=True)
       return [row.tolist() for row in mat]

# ============================== VECTOR INDEX ==============================
def build_vector_index(cfg: ECAConfig, store: Store, eb: EmbeddingBackend) -> Dict:
   chunks = store.list_chunks()
   if not chunks:
       return {"chunks": 0}
   texts = [c["text"] for c in chunks]
   vecs = eb.embed(texts)
   dim = len(vecs[0]) if vecs else 0
   if cfg.use_faiss and faiss and np is not None:
       index = faiss.IndexFlatIP(dim)
       X = np.array(vecs, dtype="float32")
       index.add(X)
       faiss.write_index(index, str(store.faiss_path))
       store.put_meta("vector_ids", {"ids": [c["chunk_id"] for c in chunks], "dim": dim, "backend": "faiss"})
       return {"chunks": len(chunks), "dim": dim, "faiss": True}
   # SQLite fallback
   store.put_meta("vectors", {"dim": dim, "backend": "sqlite",
                              "data": [{"id": c["chunk_id"], "v": v} for c, v in zip(chunks, vecs)]})
   return {"chunks": len(chunks), "dim": dim, "faiss": False}

# ============================== BM25 KEYWORD ==============================
def build_bm25(cfg: ECAConfig, store: Store) -> Dict:
   if not cfg.enable_bm25 or not BM25Okapi:
       return {"enabled": False}
   chunks = store.list_chunks()
   with store.bm25_path.open("w", encoding="utf-8") as f:
       for c in chunks:
           f.write(json.dumps({"id": c["chunk_id"], "text": c["text"]}, ensure_ascii=False) + "\n")
   store.put_meta("bm25_stats", {"N": len(chunks)})
   return {"enabled": True, "N": len(chunks)}
def _load_bm25(store: Store):
   if not BM25Okapi or not store.bm25_path.exists():
       return None, None
   docs, ids = [], []
   with store.bm25_path.open("r", encoding="utf-8") as f:
       for line in f:
           rec = json.loads(line)
           docs.append(rec["text"].split())
           ids.append(rec["id"])
   return BM25Okapi(docs), ids

# ============================== GRAPH RAG ==============================
def _extract_entities(text: str) -> List[str]:
   if _NLP:
       doc = _NLP(text)
       ents = [e.text.strip() for e in doc.ents if e.label_ in {"PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "LAW"}]
       return list({e for e in ents if len(e) > 1})
   # Fallback: conservative capitalized phrase capture
   cand = re.findall(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z0-9&\-]+){0,3})\b", text)
   return list({c for c in cand if len(c) > 1})
def _cos(a, b):
   dot = sum(x * y for x, y in zip(a, b))
   na = sqrt(sum(x * x for x in a)) + 1e-8
   nb = sqrt(sum(x * x for x in b)) + 1e-8
   return dot / (na * nb)
def build_graph(store: Store) -> Dict:
   """
   Heterogeneous graph with edges:
     - doc --contains--> chunk
     - chunk --mentions--> ENT::X
     - ENT::A --co_mention--> ENT::B
     - chunk <-> chunk (semantic similarity within same doc/order)
     - doc <-> doc (shared entities)
   Serialized as JSONL edges: {t,s,d,w}
   """
   chunks = store.list_chunks()
   edges: List[Dict] = []
   if not chunks:
       store.put_meta("graph_stats", {"edges": 0})
       with store.graph_path.open("w", encoding="utf-8") as f:
           pass
       return {"edges": 0}
   # 1) doc->chunk and chunk->entity edges
   chunk_entities: Dict[str, List[str]] = {}
   for c in chunks:
       doc_id, chunk_id, text = c["doc_id"], c["chunk_id"], c["text"]
       edges.append({"t": "contains", "s": doc_id, "d": chunk_id, "w": 1.0})
       ents = _extract_entities(text)[:50]
       chunk_entities[chunk_id] = ents
       for e in ents:
           edges.append({"t": "mentions", "s": chunk_id, "d": f"ENT::{e}", "w": 1.0})
   # 2) entity co-mentions within chunk
   for ents in chunk_entities.values():
       for i in range(len(ents)):
           for j in range(i + 1, len(ents)):
               a, b = f"ENT::{ents[i]}", f"ENT::{ents[j]}"
               edges.append({"t": "co_mention", "s": a, "d": b, "w": 1.0})
               edges.append({"t": "co_mention", "s": b, "d": a, "w": 1.0})
   # 3) chunk<->chunk similarity (using stored vectors if available)
   vecpack = store.get_meta("vectors")
   if vecpack and "data" in vecpack:
       id2vec = {row["id"]: row["v"] for row in vecpack["data"]}
       by_doc: Dict[str, List[str]] = defaultdict(list)
       for cid in id2vec.keys():
           doc_id = cid.split(":")[0]
           by_doc[doc_id].append(cid)
       for doc_id, cids in by_doc.items():
           cids_sorted = sorted(cids)
           for k in range(len(cids_sorted) - 1):
               a, b = cids_sorted[k], cids_sorted[k + 1]
               va, vb = id2vec[a], id2vec[b]
               sim = _cos(va, vb)
               if sim > 0.2:
                   edges.append({"t": "chunk_sim", "s": a, "d": b, "w": float(sim)})
                   edges.append({"t": "chunk_sim", "s": b, "d": a, "w": float(sim)})
   # 4) doc<->doc via shared entities
   doc_entities: Dict[str, set] = defaultdict(set)
   for chunk_id, ents in chunk_entities.items():
       doc_id = chunk_id.split(":")[0]
       for e in ents:
           doc_entities[doc_id].add(e)
   docs = list(doc_entities.keys())
   for i in range(len(docs)):
       for j in range(i + 1, len(docs)):
           a, b = docs[i], docs[j]
           inter = len(doc_entities[a].intersection(doc_entities[b]))
           if inter >= 2:
               w = 1.0 + 0.1 * inter
               edges.append({"t": "doc_related", "s": a, "d": b, "w": w})
               edges.append({"t": "doc_related", "s": b, "d": a, "w": w})
   with store.graph_path.open("w", encoding="utf-8") as f:
       for e in edges:
           f.write(json.dumps(e, ensure_ascii=False) + "\n")
   store.put_meta("graph_stats", {"edges": len(edges)})
   return {"edges": len(edges)}

# ----- PPR utilities -----
def _load_adj(store: Store) -> Dict[str, List[Tuple[str, float]]]:
   if not store.graph_path.exists():
       return {}
   adj: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
   with store.graph_path.open("r", encoding="utf-8") as f:
       for line in f:
           rec = json.loads(line)
           adj[rec["s"]].append((rec["d"], float(rec.get("w", 1.0))))
   return adj
def _ppr(adj: Dict[str, List[Tuple[str, float]]], seeds: List[str], alpha=0.2, iters=28) -> Dict[str, float]:
   """Personalized PageRank on directed weighted graph."""
   if not adj or not seeds:
       return {}
   # collect nodes
   nodes = set(adj.keys())
   for lst in adj.values():
       for (n, _) in lst:
           nodes.add(n)
   nodes = list(nodes)
   idx = {n: i for i, n in enumerate(nodes)}
   N = len(nodes)
   # out-weights
   outw = [0.0] * N
   for s, lst in adj.items():
       si = idx[s]
       outw[si] = sum(w for _, w in lst) or 1.0
   # seeds
   seed_idx = [idx[s] for s in seeds if s in idx]
   if not seed_idx:
       return {}
   # initialize
   p = [0.0] * N
   r = [0.0] * N
   for si in seed_idx:
       r[si] = 1.0 / len(seed_idx)
   # iterate
   for _ in range(iters):
       newp = [0.0] * N
       # distribute from p (or r in first step)
       base = p if any(p) else r
       for s, lst in adj.items():
           si = idx[s]
           mass = (1 - alpha) * base[si]
           if outw[si] == 0:
               continue
           for d, w in lst:
               di = idx[d]
               newp[di] += mass * (w / outw[si])
       # teleport to seeds
       for si in seed_idx:
           newp[si] += alpha * (1.0 / len(seed_idx))
       p = newp
   return {nodes[i]: p[i] for i in range(N) if p[i] > 0}

# ============================== SEARCH (FUSION) ==============================
def _cosine(a, b):
   dot = sum(x * y for x, y in zip(a, b))
   na = sqrt(sum(x * x for x in a)) + 1e-8
   nb = sqrt(sum(x * x for x in b)) + 1e-8
   return dot / (na * nb)
def _vsearch(cfg: ECAConfig, store: Store, eb: EmbeddingBackend, q: str, k=20) -> List[Tuple[str, float]]:
   meta = store.get_meta("vector_ids")
   if meta and faiss and store.faiss_path.exists():
       index = faiss.read_index(str(store.faiss_path))
       qv = eb.embed([q])[0]
       D, I = index.search(np.array([qv], dtype="float32"), k)
       ids = meta["ids"]
       return [(ids[i], float(d)) for d, i in zip(D[0], I[0]) if i != -1]
   vecpack = store.get_meta("vectors")
   if not vecpack:
       return []
   qv = eb.embed([q])[0]
   scored = [(row["id"], _cosine(qv, row["v"])) for row in vecpack["data"]]
   scored.sort(key=lambda x: x[1], reverse=True)
   return scored[:k]
def _bm25_search(store: Store, q: str, k=20) -> List[Tuple[str, float]]:
   bm25, ids = _load_bm25(store)
   if not bm25:
       return []
   scores = bm25.get_scores(q.split())
   ranked = sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)
   return ranked[:k]
def _extract_seed_entities(background: str, query: str) -> List[str]:
   text = (background or "").strip() + "\n" + (query or "")
   ents = _extract_entities(text)
   return [f"ENT::{e}" for e in ents]
def hybrid_search(cfg: ECAConfig, store: Store, eb: EmbeddingBackend,
                 query: str, bg: str, top_k: int = 20, graph_weight: float = 0.6) -> List[Dict]:
   """Dense + BM25 fused with Graph PPR seeded by background+query entities."""
   focus = f"{bg}\n\nQUERY: {query}".strip() if bg else query
   dense = _vsearch(cfg, store, eb, focus, k=top_k)
   kw = _bm25_search(store, focus, k=top_k) if cfg.enable_bm25 else []
   # rank fusion (Borda-ish)
   fusion: Dict[str, float] = {}
   for r, (cid, _) in enumerate(dense):
       fusion[cid] = fusion.get(cid, 0.0) + (top_k - r)
   for r, (cid, _) in enumerate(kw):
       fusion[cid] = fusion.get(cid, 0.0) + (top_k - r) * 0.8
   # graph PPR
   if cfg.enable_graph_spread:
       adj = _load_adj(store)
       seeds = _extract_seed_entities(bg, query)
       ppr = _ppr(adj, seeds, alpha=0.2, iters=28)
       # map scores to chunk nodes (direct + entity→chunk)
       graph_scores: Dict[str, float] = defaultdict(float)
       if ppr:
           for node, score in ppr.items():
               if ":" in node:  # chunk node id
                   graph_scores[node] += score
           for node, score in list(ppr.items()):
               if node.startswith("ENT::"):
                   for (nbr, w) in adj.get(node, []):
                       if ":" in nbr:
                           graph_scores[nbr] += score * 0.8
           for cid, gs in graph_scores.items():
               fusion[cid] = fusion.get(cid, 0.0) + graph_weight * gs
   # hydrate top chunks
   c = store.conn.cursor()
   out: List[Dict] = []
   for cid, _ in sorted(fusion.items(), key=lambda x: x[1], reverse=True)[:top_k]:
       c.execute("SELECT doc_id,ord,text,meta FROM chunks WHERE chunk_id=?", (cid,))
       row = c.fetchone()
       if not row:
           continue
       out.append(dict(
           chunk_id=cid,
           doc_id=row[0],
           ord=row[1],
           text=row[2],
           meta=json.loads(row[3] or "{}")
       ))
   return out

# ============================== ONE-SHOT BUILD ==============================
def build_all(cfg: ECAConfig, store: Store) -> Dict:
   eb = EmbeddingBackend(cfg)
   v = build_vector_index(cfg, store, eb)
   b = build_bm25(cfg, store)
   g = build_graph(store)
   return {"vectors": v, "bm25": b, "graph": g}