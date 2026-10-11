
from google import genai
from google.genai import types
import os

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

chat = client.chats.create(
    model="gemini-3.8-flash",
    config=types.GenerateContentConfig(
        system_instruction="You are a helpful assistant."
    )
)

while True:
    question = input("You: ")

    if question.lower() == "exit":
        break

    response = chat.send_message(question)

    print("Gemini:", response.text)
