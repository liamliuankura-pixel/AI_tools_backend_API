from typing import List, Optional, Dict
from pydantic import BaseModel, Field
class DocInput(BaseModel):
   filename: str
   text: str
class ReviewRequest(BaseModel):
   mode: str = Field(default="ensemble", description="'single' | 'ensemble' | 'collab'")
   case_background: str
   docs: List[DocInput]
   # for 'single'
   model_name: Optional[str] = None
   # for 'ensemble' / 'collab'
   model_names: Optional[List[str]] = None
class DocResult(BaseModel):
   filename: str
   final_label: str
   votes: Dict[str, str]
   rationales: Dict[str, str]
class ReviewResponse(BaseModel):
   results: List[DocResult]