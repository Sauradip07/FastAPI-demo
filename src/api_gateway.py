from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_app import rag_app  # Import the initialized RAGApp instance

app = FastAPI()

# Pydantic model for the request body
class Question(BaseModel):
    question: str

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.post("/ask")
def ask_question(question: Question):
    try:
        answer = rag_app.get_answer(question.question)
        return {"question": question.question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "API is running successfully"}
