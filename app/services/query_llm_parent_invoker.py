from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_classic.prompts import PromptTemplate
from langchain_classic.storage import LocalFileStore
from langchain_classic.storage._lc_store import create_kv_docstore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder


def query_llm_parent(query: str) -> dict:
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
    # 3. Recreate splitters exactly as used during indexing
    # -----------------------------
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

    # -----------------------------
    # 4. Load parent doc store
    # -----------------------------
    store = LocalFileStore("./got_parent_store")
    docstore = create_kv_docstore(store)

    # -----------------------------
    # 5. Parent retriever
    # -----------------------------
    parent_retriever = ParentDocumentRetriever(
        vectorstore=vector_db,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    # -----------------------------
    # 6. Retrieve parent docs
    # NOTE:
    # ParentDocumentRetriever does not directly expose child vector scores
    # here, it returns parent-level docs.
    # -----------------------------
    parent_docs = parent_retriever.invoke(query)

    print("\n--- PARENT DOCUMENTS RETRIEVED ---")
    for i, doc in enumerate(parent_docs, 1):
        print(f"\n[Parent Doc {i}]")
        print(f"Metadata: {doc.metadata}")
        print(f"Length  : {len(doc.page_content)} characters")
        print(doc.page_content[:1500].replace("\n", " "))
        print("\n" + "-" * 100)

    # -----------------------------
    # 7. Reranker
    # Rerank parent docs directly
    # -----------------------------
    reranker = CrossEncoder("BAAI/bge-reranker-base")

    pairs = [(query, doc.page_content) for doc in parent_docs]
    reranker_scores = reranker.predict(pairs)

    combined = []
    for doc, reranker_score in zip(parent_docs, reranker_scores):
        combined.append({
            "doc": doc,
            "reranker_score": float(reranker_score),
            "preview": doc.page_content[:200].replace("\n", " ")
        })

    combined_sorted = sorted(
        combined,
        key=lambda x: x["reranker_score"],
        reverse=True
    )

    # -----------------------------
    # 8. Print reranked parent docs
    # -----------------------------
    print("\n--- RERANKED PARENT DOCS ---")
    header = f"{'Rank':<6}{'RerankerScore':<18}Preview"
    print(header)
    print("-" * len(header))

    for idx, item in enumerate(combined_sorted, 1):
        print(
            f"{idx:<6}"
            f"{item['reranker_score']:<18.6f}"
            f"{item['preview']}"
        )

    # -----------------------------
    # 9. Keep top reranked parent docs
    # -----------------------------
    top_n = 3
    top_parent_docs = [item["doc"] for item in combined_sorted[:top_n]]

    print("\n--- TOP PARENT DOCS FOR FINAL CONTEXT ---")
    for i, item in enumerate(combined_sorted[:top_n], 1):
        print(f"\n[Top Parent {i}]")
        print(f"Reranker Score: {item['reranker_score']:.6f}")
        print(f"Metadata      : {item['doc'].metadata}")
        print(f"Length        : {len(item['doc'].page_content)} characters")
        print(item["doc"].page_content[:1500])
        print("\n" + "-" * 100)

    # -----------------------------
    # 10. Build context
    # -----------------------------
    context = "\n\n".join([doc.page_content for doc in top_parent_docs])

    print(f"\n--- CONTEXT LENGTH ---")
    print(f"Characters: {len(context)}")

    # -----------------------------
    # 11. LLM
    # -----------------------------
    llm = ChatOllama(model="llama3.2:3b", temperature=0)

    # -----------------------------
    # 12. Prompt
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

    return {
        "response": response.content,
        "ranked_parent_docs": [
            {
                "reranker_score": item["reranker_score"],
                "metadata": item["doc"].metadata,
                "preview": item["preview"]
            }
            for item in combined_sorted
        ]
    }
