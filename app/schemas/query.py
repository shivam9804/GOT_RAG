from typing import Dict, Any, List
from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    ranked_chunks: list
    response: str


class AgentResponseData(BaseModel):
    response: str
    status: str
    rounds: List[Dict[str, Any]]

class AgentResponseCollection(BaseModel):
    responses: Dict[str, AgentResponseData]