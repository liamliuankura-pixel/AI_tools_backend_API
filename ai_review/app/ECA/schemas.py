from pydantic import BaseModel, Field
from typing import Optional, Literal
class IndexRequest(BaseModel):
   folder: str = Field(..., description="Path to folder to index for ECA")
class QueryRequest(BaseModel):
   question: str
   mode: Optional[Literal["naive","local","global"]] = "global"
   top_k: int = 5
   model_role: Optional[Literal["base","sum","expert"]] = "base"