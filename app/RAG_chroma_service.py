import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any


class RAGServiceOLAP:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # 🔥 Embedding model (SentenceTransformers)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )

        # 🔥 Client Chroma (persistant)
        self.client = chromadb.Client(
            settings=chromadb.config.Settings(
                persist_directory="./chroma_db"
            )
        )

        # 🔥 Collection (équivalent de ton index FAISS)
        self.collection = self.client.get_or_create_collection(
            name="olap_rag",
            embedding_function=self.embedding_function
        )

    # =========================
    # 🔄 Transformer JSON OLAP → documents
    # =========================
    def _build_documents(self, cube: Dict[str, Any]) -> List[Dict[str, Any]]:
        docs = []

        cube_name = cube.get("cubeName", "")

        # 1. Measures
        for m in cube.get("measures", []):
            text = f"""
            Cube: {cube_name}
            Type: Measure
            Name: {m['name']}
            MDX: {m['mdxExpression']}
            Aliases: {', '.join(m.get('aliases', []))}
            """

            docs.append({
                "id": f"measure_{m['name']}",
                "text": text.strip(),
                "metadata": {
                    "type": "measure",
                    "name": m["name"]
                }
            })

        # 2. Dimensions
        for d in cube.get("dimensions", []):
            dim_name = d["name"]

            for h in d.get("hierarchies", []):
                text = f"""
                Cube: {cube_name}
                Type: Dimension
                Dimension: {dim_name}
                Hierarchy: {h['name']}
                MDX: {h['mdxExpression']}
                Aliases: {', '.join(h.get('aliases', []))}
                Members: {h.get('membersExpression')}
                """

                docs.append({
                    "id": f"dim_{dim_name}_{h['name']}",
                    "text": text.strip(),
                    "metadata": {
                        "type": "dimension",
                        "dimension": dim_name,
                        "name": h["name"]
                    }
                })

        # 3. Sample Queries
        for i, q in enumerate(cube.get("sampleQueries", [])):
            text = f"""
            Cube: {cube_name}
            Type: Query Example
            Question: {q['question']}
            MDX: {q['mdx']}
            """

            docs.append({
                "id": f"query_{i}",
                "text": text.strip(),
                "metadata": {
                    "type": "query",
                    "question": q["question"]
                }
            })

        return docs

    # =========================
    # 📥 Indexation (persistante 🔥)
    # =========================
    def index_cube(self, cube: Dict[str, Any]):
        docs = self._build_documents(cube)

        self.collection.add(
            ids=[d["id"] for d in docs],
            documents=[d["text"] for d in docs],
            metadatas=[d["metadata"] for d in docs]
        )

        print(f"✅ {len(docs)} documents indexés dans Chroma")

    # =========================
    # 🔍 Recherche intelligente
    # =========================
    def search(self, query: str, k: int = 5):
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )

        output = []
        for i in range(len(results["documents"][0])):
            output.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i]
            })

        return output

import requests


class BIChatbotService:
    def __init__(self, rag_service):
        self.rag = rag_service
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model = "phi3:mini"  # ou llama3

    # =========================
    # 🧠 Construire le prompt
    # =========================
    def build_prompt(self, question: str, context_docs):
        context_text = "\n\n".join([doc["text"] for doc in context_docs])

        prompt = f"""
You are an expert in OLAP and MDX query generation.

Context:
{context_text}

User Question:
{question}

Instructions:
- Generate ONLY a valid MDX query
- Use ONLY measures and dimensions from context
- Do NOT explain anything
- Do NOT add text outside MDX
- Use correct MDX syntax

MDX:
"""
        return prompt

    # =========================
    # 🤖 Appel Ollama
    # =========================
    def call_ollama(self, prompt: str):
        response = requests.post(self.ollama_url, json={
            "model": self.model,
            "prompt": prompt,
            "stream": False
        })

        return response.json()["response"]

    # =========================
    # 🔥 Pipeline complet
    # =========================
    def generate_mdx(self, question: str):
        # 1. Retrieve context
        context = self.rag.search(question, k=5)

        # 2. Build prompt
        prompt = self.build_prompt(question, context)

        # 3. Generate MDX
        mdx = self.call_ollama(prompt)

        return {
            "question": question,
            "mdx": mdx.strip(),
            "context": context
        }