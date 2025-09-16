from typing import Dict, List, Optional
from abc import ABC, abstractmethod
from .core import SYSTEM_PROMPT, USER_TEMPLATE, parse_jsonish, normalize_label, majority_vote, chunk_text
class ModelClient(ABC):
   @abstractmethod
   def classify(self, model_name: str, case_text: str, doc_text: str) -> Dict[str, str]:
       ...
   @abstractmethod
   def generate(self, model_name: str, prompt: str) -> str:
       ...
class EnsembleReviewer:
   """
   Modes:
     - single: one model, per-chunk majority
     - ensemble: N models, model-level majority
     - collab: N models propose -> critique -> arbiter final
   """
   def __init__(self, model_client: ModelClient):
       self.client = model_client
   # ---------- SINGLE ----------
   def review_doc_single(self, model_name: str, case_text: str, doc_text: str) -> Dict:
       labels, rats = [], []
       for ch in chunk_text(doc_text):
           out = self.client.classify(model_name, case_text, ch)
           labels.append(normalize_label(out["label"]))
           rats.append(out.get("rationale", ""))
       final = majority_vote(labels)
       return {"final": final, "votes": {model_name: final}, "rationales": {model_name: " | ".join(rats[:2])}}
   # ---------- ENSEMBLE ----------
   def review_doc_ensemble(self, model_names: List[str], case_text: str, doc_text: str) -> Dict:
       model_votes, model_rats = {}, {}
       for m in model_names:
           labels, rats = [], []
           for ch in chunk_text(doc_text):
               out = self.client.classify(m, case_text, ch)
               labels.append(normalize_label(out["label"]))
               rats.append(out.get("rationale", ""))
           mv = majority_vote(labels)
           model_votes[m] = mv
           model_rats[m] = " | ".join(rats[:2])
       final = majority_vote(list(model_votes.values()))
       return {"final": final, "votes": model_votes, "rationales": model_rats}
   # ---------- COLLAB ----------
   def review_doc_collab(
       self,
       model_names: List[str],
       case_text: str,
       doc_text: str,
       arbiter: Optional[str] = None,
       critique_rounds: int = 1
   ) -> Dict:
       proposals = {}
       # 1) initial proposals on full document
       for m in model_names:
           out = self.client.classify(m, case_text, doc_text)
           proposals[m] = {
               "label": normalize_label(out["label"]),
               "rationale": out.get("rationale", "")
           }
       # 2) critique rounds
       for _ in range(max(0, critique_rounds)):
           panel_text = _format_panel(case_text, doc_text, proposals)
           new_props = {}
           for m in model_names:
               crit = self.client.classify(
                   m,
                   case_text="",
                   doc_text=_CRITIQUE_PROMPT.format(panel=panel_text)
               )
               label = normalize_label(crit.get("label", proposals[m]["label"]))
               rat = crit.get("rationale", proposals[m]["rationale"])
               new_props[m] = {"label": label, "rationale": rat}
           proposals = new_props
       # 3) arbiter decision
       arbiter = arbiter or model_names[0]
       arb_in = _format_panel(case_text, doc_text, proposals)
       arb = self.client.classify(
           arbiter, case_text="", doc_text=_ARBITER_PROMPT.format(panel=arb_in)
       )
       final_label = normalize_label(arb.get("label", majority_vote([p["label"] for p in proposals.values()])))
       final_rat = arb.get("rationale", "Arbiter decision")
       return {
           "final": final_label,
           "votes": {m: p["label"] for m, p in proposals.items()},
           "rationales": {m: p["rationale"] for m, p in proposals.items()},
           "arbiter": arbiter,
           "arbiter_rationale": final_rat
       }
def _format_panel(case_text: str, doc_text: str, proposals: Dict[str, Dict[str, str]]) -> str:
   props = "\n".join([f"- {m}: {p['label']} — {p['rationale']}" for m, p in proposals.items()])
   return (
       f"CASE BACKGROUND:\n{case_text}\n\n"
       f"DOCUMENT:\n{doc_text[:4000]}\n\n"
       f"CURRENT PANEL:\n{props}\n"
   )
_CRITIQUE_PROMPT = (
   "You are collaborating with other reviewers. Given the PANEL below, respond with a JSON of your "
   "revised judgment: {\"label\":\"Responsive|Not Responsive|Needs Review\", \"rationale\":\"...\"}. "
   "If peers present strong evidence that contradicts you, adjust your label. Keep rationale concise.\n\n"
   "{panel}"
)
_ARBITER_PROMPT = (
   "Act as an impartial ARBITER. Read the PANEL and choose the most defensible final label for responsiveness. "
   "Return JSON: {\"label\":\"Responsive|Not Responsive|Needs Review\", \"rationale\":\"short reason citing evidence\"}.\n\n"
   "{panel}"
)