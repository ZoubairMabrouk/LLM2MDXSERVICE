from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any

from app.rag_service_olap import RAGServiceOLAP

import json

def load_cube_from_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

cube = load_cube_from_file("./data/metadata.json")
app = FastAPI(title="RAG OLAP Service")

rag = RAGServiceOLAP()


class CubeRequest(BaseModel):
    cube: Dict[str, Any]


class QueryRequest(BaseModel):
    query: str
    top_k: int = 10


# 🔹 Index cube
@app.on_event("startup")
def load_data():
    with open("./data/metadata.json", "r", encoding="utf-8") as f:
        cube = json.load(f)

    rag.index_cube(cube)
    print("✅ Cube chargé et indexé")


# 🔹 Search
@app.post("/search")
def search(request: QueryRequest):
    results = rag.search(request.query, request.top_k)
    return {"results": results}