from typing import Dict, List
from pydantic import BaseModel


# ------------------------------
# Request model: what the user sends to the API
# ------------------------------
class QueryRequest(BaseModel):
    # The actual question being asked
    question: str


# ------------------------------
# Response model: what our API sends back
# ------------------------------
class QueryResponse(BaseModel):
    # The LLM’s answer to the question
    answer: str

    # A list of sources that supported the answer
    # Each source is a dict like:
    #   { "source": "filename.pdf", "snippet": "Excerpt of text..." }
    sources: List[Dict[str, str]]
