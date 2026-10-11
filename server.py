
from fastapi import FastAPI
from pydantic import BaseModel
from google import genai
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://sepehrRez.github.io",
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:5501",
        "http://localhost:5501",
    ],
    allow_credentials=False,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(request: ChatRequest):
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=request.message
    )
    return {"reply": response.text}
