from typing import Dict
from ollama import chat
from ..core import SYSTEM_PROMPT, USER_TEMPLATE, parse_jsonish

class OllamaClient:
   """
   Local Ollama provider. Two operations:
     - classify(model, case_text, doc_text) -> {label, rationale}
     - generate(model, prompt) -> text
   """
   def classify(self, model_name: str, case_text: str, doc_text: str) -> Dict[str, str]:
       # If caller sends a "freeform" prompt in doc_text (for critique/arbiter), we still wrap it;
       # the JSON contract is enforced by SYSTEM_PROMPT.
       resp = chat(
           model=model_name,
           messages=[
               {"role": "system", "content": SYSTEM_PROMPT},
               {"role": "user", "content": USER_TEMPLATE.format(case=case_text, doc=doc_text)}
           ],
           think=False
       )
       content = resp.message.content
       data = parse_jsonish(content)
       return {"label": data.get("label","Needs Review"), "rationale": data.get("rationale","")}

   def generate(self, model_name: str, prompt: str) -> str:
       resp = chat(
           model=model_name,
           messages=[{"role": "user", "content": prompt}],
           think=False
       )
       return resp.message.content