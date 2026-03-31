from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_classic.agents import AgentExecutor, create_react_agent
import configparser

config = configparser.ConfigParser()
config.read('config.cfg')

from app.services.tools import (
    retrieve_context_tool,
    rewrite_query_tool,
    evaluate_evidence_tool,
)

tools = [
    retrieve_context_tool,
    rewrite_query_tool,
    evaluate_evidence_tool
]

# llm = ChatOllama(model="llama3.2:3b", temperature=0.0)
llm = ChatOllama(
    model="glm-5:cloud",
    base_url=config.get('ollama', 'BASE_URL'),
    temperature=0.0
)

# prompt = PromptTemplate.from_template("""
#     You are an evidence-based QA agent for the book "A Game of Thrones".

#     You must use the available tools to answer the user's question.

#     You have access to the following tools:
#     {tools}

#     Rules:
#     1. Use retrieve_context_tool to get evidence for the user's question.
#     2. Use evaluate_evidence_tool to check if the retrieved context is enough.
#     3. If evidence is not enough, use rewrite_query_tool to improve the query, then retrieve_context_tool again.
#     4. If evaluate_evidence_tool returns {{"enough": true, ...}}, do NOT call any more tools. Immediately provide Final Answer.
#     5. Do not answer from outside knowledge. If no enough evidence exists, say: "I don't know based on the provided context."

#     Use this format:

#     Question: the input question you must answer
#     Thought: think about what to do next
#     Action: the action to take, must be EXACTLY one of [{tool_names}]
#     Action Input: the input to the action
#     Observation: the result of the action
#     ... (repeat Thought/Action/Action Input/Observation as needed)
#     Thought: I now know the final answer
#     Final Answer: the final answer to the original input question

#     Example:

#     Question: Who is Arya Stark?
#     Thought: I need to retrieve context about Arya Stark first.
#     Action: retrieve_context_tool
#     Action Input: Arya Stark
#     Observation: Arya Stark is ...

#     Thought: Now I should evaluate if this context is enough to answer the question.
#     Action: evaluate_evidence_tool
#     Action Input: QUESTION: Who is Arya Stark?
#     CONTEXT: Arya Stark is ...
#     Observation: {{"enough": false, "reason": "Context is incomplete, missing character details"}}

#     Thought: The evidence is not enough. I should rewrite the query to get better results.
#     Action: rewrite_query_tool
#     Action Input: Who is Arya Stark?
#     Observation: Arya Stark character background and role

#     Thought: Now let me retrieve context again with the improved query.
#     Action: retrieve_context_tool
#     Action Input: Arya Stark character background and role
#     Observation: Arya Stark is the younger daughter ...

#     Thought: Let me evaluate this new context.
#     Action: evaluate_evidence_tool
#     Action Input: QUESTION: Who is Arya Stark?
#     CONTEXT: Arya Stark is the younger daughter ...
#     Observation: {{"enough": true, "reason": "Context contains enough information about Arya Stark's background"}}

#     Thought: I now have enough evidence to answer the question.
#     Final Answer: Arya Stark is a major character in A Game of Thrones...

#     Question: {input}
#     Thought: {agent_scratchpad}
#     """
# )

prompt = PromptTemplate.from_template("""
    You are an evidence-based QA agent for the book "A Game of Thrones".

    You must use the available tools to answer the user's question.

    You have access to the following tools:
    {tools}

    Rules:
    1. Use retrieve_context_tool to get evidence for the user's question.
    2. Use evaluate_evidence_tool to check if the retrieved context is enough.
    3. If evidence is not enough, use rewrite_query_tool to improve the query, then retrieve_context_tool again.
    4. If evaluate_evidence_tool returns {{"enough": true, ...}}, immediately provide Final Answer.
    5. Do not answer from outside knowledge. If no enough evidence exists, say: "I don't know based on the provided context."

    Important:
    - Output either:
    a) Thought + Action + Action Input
    OR
    b) Thought + Final Answer
    - Never output Observation yourself.
    - Never output both an Action and a Final Answer in the same response.

    Use this format:

    Question: the input question you must answer
    Thought: think about what to do next
    Action: the action to take, must be EXACTLY one of [{tool_names}]
    Action Input: the input to the action

    OR

    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

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
    handle_parsing_errors=False,
    return_intermediate_steps=True
)

def invoke_agent(query: str) -> str:
    response = agent_executor.invoke({"input": query})

    print(response.keys())
    print(type(response))
    print(f"Agent Response: {response}")

    return response.get("output", "No response generated")
    

