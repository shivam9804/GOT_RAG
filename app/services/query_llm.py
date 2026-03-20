from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_classic.prompts import PromptTemplate
from sentence_transformers import CrossEncoder

def query_llm(query: str) -> str:
    # Placeholder for LLM invocation logic
    # In a real implementation, this would call the LLM API and return the response

    # -----------------------------
    # 1. Embeddings
    # -----------------------------
    embeddings = OllamaEmbeddings(model="nomic-embed-text")


    # -----------------------------
    # 2. Load existing vector DB
    # -----------------------------
    vector_db = Chroma(
        collection_name="got_parent_child",
        persist_directory="./got_parent_child_chroma",
        embedding_function=embeddings
    )

    # -----------------------------
    # 3. Initial retrieval with vector scores
    # NOTE:
    # For Chroma, lower distance is usually better.
    # If using cosine distance, smaller = more similar.
    # -----------------------------
    k_initial = 12
    retrieved_with_scores = vector_db.similarity_search_with_score(query, k=k_initial)

    print("\n--- INITIAL RETRIEVAL WITH VECTOR SCORES ---")
    for i, (doc, score) in enumerate(retrieved_with_scores, 1):
        print(f"\n[Chunk {i}] Vector Score: {score:.6f}")
        print(f"Metadata: {doc.metadata}")
        print(doc.page_content[:700].replace("\n", " "))
        print("\n" + "-" * 100)


    # -----------------------------
    # 4. Reranker
    # First run downloads automatically from Hugging Face
    # -----------------------------
    reranker = CrossEncoder("BAAI/bge-reranker-base")

    docs = [doc for doc, _ in retrieved_with_scores]
    pairs = [(query, doc.page_content) for doc in docs]
    reranker_scores = reranker.predict(pairs)

    # Combine everything into one structure
    combined = []
    for (doc, vector_score), reranker_score in zip(retrieved_with_scores, reranker_scores):
        combined.append({
            "doc": doc,
            "vector_score": float(vector_score),
            "reranker_score": float(reranker_score),
            "preview": doc.page_content[:160].replace("\n", " ")
        })

    # For Chroma distance: lower vector_score is better
    # For reranker: higher score is better
    combined_sorted = sorted(combined, key=lambda x: x["reranker_score"], reverse=True)


    # -----------------------------
    # 5. Print debug ranking table
    # -----------------------------
    print("\n--- DEBUG RANKING TABLE ---")
    header = f"{'Rank':<6}{'VectorScore':<16}{'RerankerScore':<18}Preview"
    print(header)
    print("-" * len(header))

    for idx, item in enumerate(combined_sorted, 1):
        print(
            f"{idx:<6}"
            f"{item['vector_score']:<16.6f}"
            f"{item['reranker_score']:<18.6f}"
            f"{item['preview']}"
        )


    # -----------------------------
    # 6. Keep top reranked docs
    # -----------------------------
    top_n = 3
    reranked_docs = [item["doc"] for item in combined_sorted[:top_n]]

    print("\n--- TOP RERANKED CHUNKS ---")
    for i, item in enumerate(combined_sorted[:top_n], 1):
        print(f"\n[Top {i}]")
        print(f"Vector Score  : {item['vector_score']:.6f}")
        print(f"Reranker Score: {item['reranker_score']:.6f}")
        print(f"Metadata      : {item['doc'].metadata}")
        print(item["doc"].page_content[:1000])
        print("\n" + "-" * 100)


    # -----------------------------
    # 7. Build context
    # -----------------------------
    context = "\n\n".join([doc.page_content for doc in reranked_docs])

    print(f"\n--- CONTEXT LENGTH ---\nCharacters: {len(context)}")


    # -----------------------------
    # 8. LLM
    # -----------------------------
    llm = ChatOllama(model="llama3.2:3b", temperature=0)


    # -----------------------------
    # 9. Prompt
    # -----------------------------
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

    print("\n--- QUESTION ---")
    print(query)
    print("\n--- ANSWER ---")
    print(response.content)

    return {"response": response.content, "ranked_chunks": combined_sorted}
