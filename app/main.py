from fastapi import FastAPI, HTTPException
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
class BaseLLMClient:
    def __init__(self, model: str = "phi3:mini", temperature: float = 0.1):
        self._model = model
        self._temperature = temperature
        self._client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )

    def call(self, prompt: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an OLAP expert. "
                            "Generate ONLY a valid MDX query. "
                            "Do NOT explain. Do NOT add text. ONLY return MDX."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=200
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            raise RuntimeError(f"LLM API call failed: {e}")


# =========================
# FASTAPI INIT
# =========================
app = FastAPI(title="LLM + RAG API", version="2.0")

client = BaseLLMClient()


# =========================
# STARTUP → LOAD RAG
# =========================
@app.on_event("startup")
def startup():
    cube = load_cube()
    rag.index_cube(cube)
    print("✅ RAG ready")


# =========================
# REQUEST / RESPONSE
# =========================
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
        rag_results = rag.search(request.prompt, request.top_k)

        if not rag_results:
            raise HTTPException(status_code=400, detail="No context found")

        # 🧠 Step 2: Build enriched prompt
        prompt = build_rag_prompt(request.prompt, rag_results)

        # 🤖 Step 3: LLM call
        mdx_query = client.call(prompt)

        if not mdx_query:
            raise HTTPException(status_code=500, detail="Empty LLM response")

        return {"mdx": mdx_query}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))