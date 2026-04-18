from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from openai import OpenAI
from rag_service_olap import RAGServiceOLAP
import json


# =========================
# LOAD RAG
# =========================
def load_cube():
    with open("./data/metadata.json", "r", encoding="utf-8") as f:
        return json.load(f)


rag = RAGServiceOLAP()


# =========================
# LLM CLIENT
# =========================
import requests


class BaseLLMClient:
    def __init__(self, model: str = "phi3:mini", temperature: float = 0.0):
        self._model = model
        self._temperature = temperature
        self._url = "http://51.68.44.166:11434/api/chat"

    def call(self, prompt: str) -> str:
        try:
            print("🚀 Sending request to Ollama...")

            response = requests.post(
                self._url,
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": "Generate ONLY MDX."},
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False,
                    "keep_alive": "15m"
                },
                timeout=120
            )

            print("📥 Response received from Ollama")

            response.raise_for_status()

            data = response.json()

            return data["message"]["content"].strip()

        except Exception as e:
            print("❌ ERROR:", str(e))
            raise RuntimeError(f"LLM API call failed: {e}")

# =========================
# FASTAPI INIT
# =========================
app = FastAPI(title="LLM + RAG API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


client = BaseLLMClient()


# =========================
# STARTUP → LOAD RAG
# =========================
@app.on_event("startup")
def startup():
    cube = load_cube()
    rag.index_cube(cube)
    print("✅ RAG ready")



class PromptRequest(BaseModel):
    prompt: str
    top_k: Optional[int] = 10


class PromptResponse(BaseModel):
    mdx: str


# =========================
# PROMPT BUILDER 🔥
# =========================
def build_rag_prompt(user_query: str, context_docs: list) -> str:
    context = "\n\n".join([doc["text"] for doc in context_docs])

    return f"""
You are an OLAP expert working with an OLAP cube.

Here is the relevant cube metadata:
{context}

User question:
{user_query}

Task:
Generate a valid MDX query that answers the question from the context.

Rules:
- ONLY return MDX
- No explanation
- No comments
- No markdown
- No text before or after
"""


# =========================
# ENDPOINT 🔥
# =========================
@app.post("/ask", response_model=PromptResponse)
def ask(request: PromptRequest):
    try:
        # 🔍 Step 1: RAG
        i = datetime.now().isoformat()
        print(f"[{i}] 🔍 Searching RAG for: {request.prompt}")
        rag_results = rag.search(request.prompt, request.top_k)

        if not rag_results:
            raise HTTPException(status_code=400, detail="No context found")

        # 🧠 Step 2: Build enriched prompt
        i = datetime.now().isoformat()
        print(f"[{i}] 🧠 Building enriched prompt")
        prompt = build_rag_prompt(request.prompt, rag_results)

        # 🤖 Step 3: LLM call
        i = datetime.now().isoformat()
        print(f"[{i}] 🤖 Calling LLM")
        mdx_query = client.call(prompt)
        i = datetime.now().isoformat()
        print(f"[{i}] ✅ LLM response received")
        if not mdx_query:
            raise HTTPException(status_code=500, detail="Empty LLM response")

        return {"mdx": mdx_query}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))