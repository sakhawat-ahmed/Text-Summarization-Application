from fastapi import FastAPI
from pydantic import BaseModel
import nltk
from summarizer import TextSummarizer
from vector_store import VectorStore

nltk.data.path.append("backend/nltk_data")

app = FastAPI()

# -------------------- INITIALIZATIONS --------------------
summarizer = TextSummarizer()
vector_db = VectorStore()

# Summary history storage
summary_history = []

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
    # Generate summary
    summary = summarizer.summarize(req.text)

    # Save to vector DB
    vector_db.add(req.text, summary)

    # Save to summary history
    summary_history.append({
        "input": req.text,
        "summary": summary
    })

    return {"summary": summary}


@app.post("/search")
def semantic_search(req: SearchQuery):
    results = vector_db.search(req.query)
    return {"results": results}


# -------------------- HISTORY ENDPOINT --------------------
@app.get("/history")
def get_history():
    return summary_history


# -------------------- RUN SERVER --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
