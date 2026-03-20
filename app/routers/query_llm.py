from fastapi import APIRouter
from app.schemas.query import QueryRequest, QueryResponse
from app.services.query_llm import query_llm
from app.services.query_llm_parent_invoker import query_llm_parent

router = APIRouter(prefix="/query", tags=["query"])

@router.post("/", response_model=QueryResponse)
def read_query(query: QueryRequest):
    result = query_llm_parent(query.query)
    return QueryResponse(
        response=result["response"],
        ranked_chunks=result["ranked_parent_docs"]
    )
