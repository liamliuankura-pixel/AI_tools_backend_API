from typing import List, Dict, Tuple
import re, json

SYSTEM_PROMPT = (
   "You are an eDiscovery responsiveness reviewer.\n"
   "Decide if the provided document is RESPONSIVE to the case background.\n"
   "Definitions:\n"
   "- Responsive: The document materially relates to the issues described in the background.\n"
   "- Not Responsive: It does not materially relate.\n"
   "- Needs Review: The content is unclear/insufficient or borderline; a human should check.\n\n"
   "Output only valid JSON with keys: label ('Responsive'|'Not Responsive'|'Needs Review') and rationale (short explanation)."
)
USER_TEMPLATE = "CASE BACKGROUND:\n{case}\n\nDOCUMENT:\n{doc}"
MAX_DOC_CHARS = 100_000
CHUNK_CHARS = 6_000

def chunk_text(text: str, chunk: int = CHUNK_CHARS) -> List[str]:
   text = (text or "")[:MAX_DOC_CHARS]
   return [text[i:i+chunk] for i in range(0, len(text), chunk)] or [""]

def parse_jsonish(s: str) -> Dict:
   m=re.sub(r"<think>[\s\S]*?</think>", "", s, flags=re.IGNORECASE)
   m = re.search(r"\{.*\}", m, re.DOTALL)
   try:
       return json.loads(m.group(0) if m else s)
   except Exception:
       return {"label": "Needs Review", "rationale": f"Parse error: {s[:200]}"}
   
def normalize_label(label: str) -> str:
   low = (label or "").strip().lower()
   if low in {"responsive", "relevant"}:
       return "Responsive"
   if low in {"not responsive", "non-responsive", "irrelevant", "not relevant"}:
       return "Not Responsive"
   return "Needs Review"

def majority_vote(labels: List[str]) -> str:
   counts = {"Responsive": 0, "Not Responsive": 0, "Needs Review": 0}
   for l in labels:
       counts[l if l in counts else "Needs Review"] += 1
   best = max(counts.items(), key=lambda kv: kv[1])
   return "Needs Review" if list(counts.values()).count(best[1]) > 1 else best[0]