from fastapi import FastAPI
from pydantic import BaseModel

from answer import answer

app = FastAPI()

class Question(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "RAG assistant is running"}


@app.post("/ask")
def ask(body: Question):
    text, top = answer(body.question)
    return {
        "answer": text,
        "sources": [
            {"source": c["source"], "score": round(float(score), 3)}
            for score, c in top
        ],
    }