from typing import Dict, List
from pydantic import BaseModel



class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, str]]