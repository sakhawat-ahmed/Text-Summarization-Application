from fastapi import FastAPI
from pydantic import BaseModel
import nltk
from summarizer import TextSummarizer
from vector_store import VectorStore
import nltk


nltk.data.path.append("backend/nltk_data")


app = FastAPI()

summarizer = TextSummarizer()
vector_db = VectorStore()

# -------------------- MODELS --------------------

class SummarizeRequest(BaseModel):
    text: str

class SearchQuery(BaseModel):
    query: str

# -------------------- ROUTES --------------------

@app.get("/")
def root():
    return {"status": "Backend Running", "message": "Text Summarization API"}

@app.post("/summarize")
def summarize_text(req: SummarizeRequest):
    summary = summarizer.summarize(req.text)
    vector_db.add(req.text, summary)
    return {"summary": summary}

@app.post("/search")
def semantic_search(req: SearchQuery):
    results = vector_db.search(req.query)
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

