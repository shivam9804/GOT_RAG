from fastapi import APIRouter
from app.schemas.query import QueryRequest, QueryResponse, AgentResponseData, AgentResponseCollection
from app.services.query_llm import query_llm
from app.services.query_llm_parent_invoker import query_llm_parent
from app.services.query_llm_agent import query_agent
from app.services.query_llm_agent_tool import invoke_agent

router = APIRouter(prefix="/query", tags=["query"])

@router.post("/", response_model=QueryResponse)
def read_query(query: QueryRequest):
    result = query_llm_parent(query.query)
    return QueryResponse(
        response=result["response"],
        ranked_chunks=result["ranked_parent_docs"]
    )

@router.post("/agent", response_model=AgentResponseCollection)
def read_query(query: QueryRequest):
    result = query_agent(query.query)
    
    agent_response_data = AgentResponseData(
        response=result.get("response", {}),
        status=result.get("status", "Unknown"),
        rounds=result.get("rounds", [])
    )

    response_dict = {
        "agent_response": agent_response_data  # Use a meaningful string key
    }

    return AgentResponseCollection(responses=response_dict)

@router.post("/tools", response_model=AgentResponseCollection)
def read_query(query: QueryRequest):
    result = invoke_agent(query.query)
    
    agent_response_data = AgentResponseData(
        response=result.get("response", {}),
        status=result.get("status", "Unknown"),
        rounds=result.get("rounds", [])
    )

    response_dict = {
        "agent_response": agent_response_data  # Use a meaningful string key
    }

    return AgentResponseCollection(responses=response_dict)
