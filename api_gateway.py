# api_gateway.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_app import rag_app  # Import the initialized RAGApp instance

app = FastAPI()

# Pydantic model for the request body
class Question(BaseModel):
    question: str

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

# To run the FastAPI app:
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
