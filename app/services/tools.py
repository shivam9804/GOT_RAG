import json
from langchain_core.tools import tool
from app.services.query_llm_agent import (
    retrieve_context,
    rewrite_query,
    evaluate_evidence
)

@tool
def retrieve_context_tool(query: str) -> str:
    """ Retrieve relevant context from the book A Game of Thrones for a given question"""
    result = retrieve_context(query, top_n=3)
    res = evaluate_evidence(query, result["context"])
    return res

# @tool
# def evaluate_evidence_tool(query: str, context: str) -> str:
#     """Evaluate whether the retrieved context is sufficient to answer the question."""
#     print(f"""evaluate_evidence_tool: {query}""", query)
#     result = evaluate_evidence(query, context)
#     return json.dumps(result)

@tool
def rewrite_query_tool(query: str) -> str:
    """Rewrite a question to improve retrieval for a book QA system."""
    result = rewrite_query(query, "Evidence is weak, partial, or only loosely related.")
    return result
