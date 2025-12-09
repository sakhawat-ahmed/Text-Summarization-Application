import chromadb
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection("summaries")
        self.embedder = SentenceTransformer("/app/local_models/all-MiniLM-L6-v2")



    def add(self, text, summary):
        embedding = self.embedder.encode(summary).tolist()
        self.collection.add(
            ids=[str(hash(text))],
            documents=[summary],
            embeddings=[embedding],
            metadatas=[{"original": text}]
        )

    def search(self, query):
        embedding = self.embedder.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=5
        )
        return results
