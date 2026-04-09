import faiss
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer


class RAGServiceOLAP:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self.metadata_store = []  # 🔥 important pour garder info structurée

    # =========================
    # 🔄 Transformer JSON OLAP → documents
    # =========================
    def _build_documents(self, cube: Dict[str, Any]) -> List[Dict[str, Any]]:
        docs = []

        cube_name = cube.get("cubeName", "")

        # =====================
        # 1. Measures
        # =====================
        for m in cube.get("measures", []):
            text = f"""
            Cube: {cube_name}
            Type: Measure
            Name: {m['name']}
            MDX: {m['mdxExpression']}
            Aliases: {', '.join(m.get('aliases', []))}
            """
            docs.append({
                "text": text.strip(),
                "type": "measure",
                "name": m["name"]
            })

        # =====================
        # 2. Dimensions + Hierarchies
        # =====================
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
                    "text": text.strip(),
                    "type": "dimension",
                    "name": h["name"]
                })

        # =====================
        # 3. Sample Queries (🔥 très puissant)
        # =====================
        for q in cube.get("sampleQueries", []):
            text = f"""
            Cube: {cube_name}
            Type: Query Example
            Question: {q['question']}
            MDX: {q['mdx']}
            """
            docs.append({
                "text": text.strip(),
                "type": "query",
                "question": q["question"]
            })

        return docs

    # =========================
    # 📥 Indexation
    # =========================
    def index_cube(self, cube: Dict[str, Any]):
        docs = self._build_documents(cube)

        texts = [d["text"] for d in docs]
        embeddings = self.model.encode(texts)

        dim = embeddings.shape[1]

        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)

        self.index.add(np.array(embeddings).astype("float32"))

        self.documents.extend(texts)
        self.metadata_store.extend(docs)

    # =========================
    # 🔍 Recherche intelligente
    # =========================
    def search(self, query: str, k: int = 10):
        query_embedding = self.model.encode([query])

        distances, indices = self.index.search(
            np.array(query_embedding).astype("float32"), k
        )

        results = []
        for idx in indices[0]:
            if idx < len(self.documents):
                results.append({
                    "text": self.documents[idx],
                    "meta": self.metadata_store[idx]
                })

        return results