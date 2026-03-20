from fastapi import FastAPI
from app.routers import query_llm, vector_db

app = FastAPI()

app.include_router(query_llm.router)
app.include_router(vector_db.router)

@app.get("/")
def read_root():
    return {"message": "Hello from project-2!"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}

