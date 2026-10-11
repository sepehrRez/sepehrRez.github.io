
from fastapi import FastAPI
from pydantic import BaseModel
from google import genai
import os
from fastapi.middleware.cors import CORSMiddleware
from AiBot.rag import search_resume
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
def chat(request: ChatRequest):
    try:
        # Find relevant information in the resume
        chunks = search_resume(request.message)
        context = "\n\n".join(chunks)

        prompt = f"""
You are Sepehr's portfolio assistant.
Answer questions using the resume context below.

Rules:
- Use only information supported by the context.
- If the answer is not in the context, say you couldn't find
  that information in Sepehr's resume.
- Do not invent facts.

Resume context:
{context}

Question: {request.message}
"""

        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt
        )

        return {"reply": response.text}

    except Exception as e:
        print("Chat error:", repr(e))
        return {"error": "Sorry, something went wrong."}

