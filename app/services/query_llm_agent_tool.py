from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_classic.agents import AgentExecutor, create_react_agent

from app.services.tools import (
    retrieve_context_tool,
    rewrite_query_tool,
    # evaluate_evidence_tool,
)

tools = [
    retrieve_context_tool,
    rewrite_query_tool,
]

llm = ChatOllama(model="llama3.2:3b", temperature=0.0)

prompt = PromptTemplate.from_template("""
    You are an evidence-based QA agent for the book "A Game of Thrones".

    You must use the available tools to answer the user's question.

    You have access to the following tools:\n{tools}

    Rules:
    1. First use retrieve_context_tool to get evidence for the user's question.
    2. If the retrieved context is insufficient, use rewrite_query_tool to improve the query, then retrieve_context_tool again.
    3. Do not answer from outside knowledge.
    4. If after trying you still do not have enough evidence, say:
    "I don't know based on the provided context."

    Use this format:

    Question: the input question you must answer
    Thought: think about what to do next
    Action: the action to take, must be EXACTLY one of [{tool_names}]
    Action Input: the input to the action as plain text
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Example:

    Question: Who is Arya Stark?
    Thought: I should first retrieve evidence.
    Action: retrieve_context_tool
    Action Input: Arya Stark
    Observation: "<context snippet about Arya Stark>"

    Thought: I should rewrite the query and try again.
    Action: rewrite_query_tool
    Action Input: Arya Stark

    Observation: Arya Stark character description

    Thought: I should retrieve evidence again using the improved query.
    Action: retrieve_context_tool
    Action Input: Arya Stark character description
    Observation: "<context snippet that contains direct evidence>"

    Thought: I now know the final answer
    Final Answer: Arya Stark is ...

    Question: {input}
    Thought: {agent_scratchpad}
    """
)

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True
)

def invoke_agent(query: str) -> str:
    response = agent_executor.invoke({"input": query})
    if response:
        return response["output"]
    else:
        response
