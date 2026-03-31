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
    return result['context']

@tool
def evaluate_evidence_tool(input_text: str) -> str:
    """
    Evaluate whether the retrieved context is enough to answer the question.
    
    Expected input format:
    QUESTION: <question>
    CONTEXT: <retrieved context>
    """

    try:
        question_part = input_text.split("CONTEXT:")[0].replace("QUESTION:", "").strip()
        context_part = input_text.split("CONTEXT:")[1].strip()
    except IndexError:
        return json.dumps({
            "enough": False,
            "reason": "Input format invalid. Expected QUESTION: ... CONTEXT: ..."
        })

    result = evaluate_evidence(question_part, context_part)
    return json.dumps(result)

@tool
def rewrite_query_tool(query: str) -> str:
    """Rewrite a question to improve retrieval for a book QA system."""
    result = rewrite_query(query, "Evidence is weak, partial, or only loosely related.")
    return result
