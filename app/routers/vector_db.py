from fastapi import APIRouter
from app.schemas.query import QueryRequest, QueryResponse
from app.services.query_llm import query_llm
from app.services.db_main import create_got_knowledge_base

router = APIRouter(prefix="/vector-db", tags=["vector-db"])

@router.get("/")
def read_query():
    return create_got_knowledge_base()
