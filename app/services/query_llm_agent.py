from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_classic.prompts import PromptTemplate
from langchain_classic.storage import LocalFileStore
from langchain_classic.storage._lc_store import create_kv_docstore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

# -----------------------------
# Shared setuo
# -----------------------------

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_db = Chroma(
    collection_name="got_parent_child",
    persist_directory="./got_parent_child_chroma",
    embedding_function=embeddings
)

parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1800,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", ", ", " ", ""]
)

child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", ", ", " ", ""]
)

store = LocalFileStore("./got_parent_store")
docstore = create_kv_docstore(store)

parent_retriever = ParentDocumentRetriever(
    vectorstore=vector_db,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_kwargs={"k": 12}
)

reranker = CrossEncoder("BAAI/bge-reranker-base")
llm = ChatOllama(model="llama3.2:3b", temperature=0)

# -----------------------------
# 1. Retrieval loop
# -----------------------------
def retrieve_context(query: str, top_n: int = 3) -> dict:
    parent_docs = parent_retriever.invoke(query)

    pairs = [(query, doc.page_content) for doc in parent_docs]
    scores = reranker.predict(pairs) if pairs else []

    combined = []
    for (doc, score) in zip(parent_docs, scores):
        combined.append({
            "doc": doc,
            "reranker_score": float(score),
            "preview": doc.page_content[:200].replace("\n", " ")
        })

    combined_sorted = sorted(
        combined,
        key=lambda x: x['reranker_score'],
        reverse=True
    )

    top_docs = [item["doc"] for item in combined_sorted[:top_n]]
    context = "\n\n".join(doc.page_content for doc in top_docs)

    return {
        "context": context,
        "ranked_docs": combined_sorted,
        "top_docs": top_docs
    }

# -----------------------------
# 2. Evidence evaluation step
# -----------------------------
def evaluate_evidence(query: str, context: str) -> dict:
    template = """
        You are checking whether the retrieved context is sufficient to answer the question.

        Rules:
        1. Judge only based on the provided context.
        2. Decide if the context contains enough direct evidence.
        3. If evidence is weak, partial, or only loosely related, mark it as insufficient.
        4. If a direct supporting quote exists, include it.
        5. Return only valid JSON.

        Return exactly this JSON format:
        {{
            "enough": true or false,
            "reason": "<brief explanation>",
            "quote": "<direct supporting quote>" or ""
        }}

        Context:
        {context}

        Question:
        {question}
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    final_prompt = prompt.format(context=context, question=query)
    response = llm.invoke(final_prompt)

    import json
    try:
        return json.loads(response.content)
    except json.decoder.JSONDecodeError:
        return {
            "enough": False,
            "reason": "Unable to parse JSON",
            "quote": ""
        }

# -----------------------------
# 2. Query rewrite step
# -----------------------------
def rewrite_query(query: str, previous_reason: str) -> str:
    template = """
        You are rewriting a question to improve retrieval for a book QA system.

        Rules:
        1. Preserve the original meaning.
        2. Make the query more searchable.
        3. Prefer concrete character names, events, and relationships if relevant.
        4. Do not answer the question.
        5. Return only the rewritten query.

        Original question:
        {question}

        Why previous retrieval was insufficient:
        {reason}
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["question", "reason"]    
    )

    final_prompt = prompt.format(question=query, reason=previous_reason)
    response = llm.invoke(final_prompt)

    return response.content.strip()


# -----------------------------
# 4. Final answer generation
# -----------------------------
def generate_answer(query: str, context: str) -> str:
    template = """
        You are answering questions only from the provided context from the book 'A Game of Thrones'.

        Rules:
        1. Answer only from the provided context.
        2. Do not use outside knowledge.
        3. If the context gives only partial evidence, say that the answer is only partially supported.
        4. If the answer is not clearly supported, say exactly: "I don't know based on the provided context."
        5. Do not guess names, relationships, or intentions.
        6. Keep the answer concise and specific.
        7. The Evidence quote must directly support the Answer.
        8. If no direct supporting quote exists in the provided context, say exactly:
        "No direct quote supporting the answer was found in the retrieved context."

        Return your response in exactly this format:

        Answer:
        <your answer>

        Evidence:
        "<direct quote from the context>" OR "No direct quote supporting the answer was found in the retrieved context."

        Context:
        {context}

        Question:
        {question}
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    final_prompt = prompt.format(context=context, question=query)
    response = llm.invoke(final_prompt)

    return response.content


# -----------------------------
# 5. Agent loop
# -----------------------------
def query_agent(query: str, max_round: int = 2) -> str:

    current_query = query
    rounds = []

    for step in range(max_round):
        retrival = retrieve_context(current_query, top_n=3)
        evaluation = evaluate_evidence(query, retrival["context"])
        
        rounds.append({
            "step": step + 1,
            "query_used": current_query,
            "evaluation": evaluation,
            "ranked_docs": [
                {
                    "reranker_score": item["reranker_score"],
                    "metadata": item["doc"].metadata,
                    "preview": item["preview"]
                }
                for item in retrival["ranked_docs"]
            ],
        })
        
        if evaluation.get("enough") is True:
            answer = generate_answer(query, retrival["context"])
            return {
                "response": answer,
                "status": "answered",
                "rounds": rounds
            }
        
        current_query = rewrite_query(
            query=current_query,
            previous_reason=evaluation.get("reason", "Evidence was insufficient.")
        )

    # Final fallback after retries
    final_retrival = retrieve_context(current_query, top_n=3)
    final_answer = generate_answer(query, final_retrival["context"])

    return {
        "response": final_answer,
        "status": "answered",
        "rounds": rounds
    }
