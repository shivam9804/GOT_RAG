import os
import re
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from langchain_classic.storage import LocalFileStore
from langchain_classic.storage._lc_store import create_kv_docstore
from langchain_classic.retrievers import ParentDocumentRetriever


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("\x00", " ")
    text = text.replace("\t", " ")
    text = text.replace("\r", " ")
    text = text.strip()
    return text

def create_got_knowledge_base():
    # -----------------------------
    # 1. Load the Game of Thrones PDF
    # -----------------------------
    file_path = "./app/books/1-A Game of Thrones.pdf"
    loader = PyPDFLoader(file_path)
    docs = loader.load()


    # -----------------------------
    # 2. Light cleanup of PDF text
    # -----------------------------
    cleaned_docs = []
    for i, doc in enumerate(docs):
        doc.page_content = clean_text(doc.page_content)
        doc.metadata["page_number"] = i + 1
        cleaned_docs.append(doc)


    # -----------------------------
    # 3. Parent and child splitters
    # -----------------------------
    # Parent chunks = larger context blocks
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1800,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", ", ", " ", ""]
    )

    # Child chunks = smaller searchable units
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", ", ", " ", ""]
    )


    # -----------------------------
    # 4. Embeddings
    # -----------------------------
    embeddings = OllamaEmbeddings(model="nomic-embed-text")


    # -----------------------------
    # 5. Persistent vector DB
    # -----------------------------
    persist_directory = "./got_parent_child_chroma"

    # Optional: reset DB while experimenting
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)

    if os.path.exists("./got_parent_store"):
        shutil.rmtree("./got_parent_store")

    vectorstore = Chroma(
        collection_name="got_parent_child",
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )


    # -----------------------------
    # 6. Parent doc store on disk
    # -----------------------------
    store = LocalFileStore("./got_parent_store")
    docstore = create_kv_docstore(store)


    # -----------------------------
    # 7. ParentDocumentRetriever
    # -----------------------------
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )


    # -----------------------------
    # 8. Add documents
    # -----------------------------
    retriever.add_documents(cleaned_docs)


    print("Success! Parent-child GOT knowledge base created.")
    print(f"Vector store path: {persist_directory}")
    print("Parent docs stored in: ./got_parent_store")

    return {"status": "success", "message": "GOT knowledge base created."}
