
import os
from pathlib import Path

from google import genai
from google.genai import types
from pypdf import PdfReader
import chromadb


BASE_DIR = Path(__file__).resolve().parent
RESUME_PATH = BASE_DIR / "resume.pdf"
DB_PATH = BASE_DIR / "chroma_db"

embedding_client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

chroma_client = chromadb.PersistentClient(
    path=str(DB_PATH)
)

# Use a new collection because the embedding dimensions changed.
collection = chroma_client.get_or_create_collection(
    name="resume_gemini",
    metadata={"hnsw:space": "cosine"}
)


def extract_text(pdf_path):
    reader = PdfReader(str(pdf_path))
    pages = []

    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)

    return "\n".join(pages)


def split_into_chunks(text, chunk_size=120, overlap=30):
    words = text.split()
    chunks = []

    for start in range(0, len(words), chunk_size - overlap):
        chunk = words[start:start + chunk_size]

        if not chunk:
            break

        chunks.append(" ".join(chunk))

        if start + chunk_size >= len(words):
            break

    return chunks


def embed_texts(texts, task_type):
    vectors = []

    # Process batches to avoid oversized requests.
    for start in range(0, len(texts), 50):
        batch = texts[start:start + 50]

        response = embedding_client.models.embed_content(
            model="gemini-embedding-001",
            contents=batch,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=768
            )
        )

        vectors.extend(
            embedding.values
            for embedding in response.embeddings
        )

    return vectors


def index_resume():
    if not RESUME_PATH.exists():
        raise FileNotFoundError(
            f"Resume not found: {RESUME_PATH}"
        )

    text = extract_text(RESUME_PATH)

    if not text.strip():
        raise ValueError(
            "No text extracted from the resume PDF."
        )

    chunks = split_into_chunks(text)

    if not chunks:
        raise ValueError("No resume chunks were created.")

    embeddings = embed_texts(
        chunks,
        "RETRIEVAL_DOCUMENT"
    )

    existing = collection.get()

    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    collection.add(
        ids=[f"chunk-{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings
    )

    print(f"Indexed {len(chunks)} chunks.")


def search_resume(question, top_k=5):
    count = collection.count()

    if count == 0:
        raise RuntimeError(
            "Resume is not indexed."
        )

    query_embedding = embed_texts(
        [question],
        "RETRIEVAL_QUERY"
    )[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, count)
    )

    return results["documents"][0]


if __name__ == "__main__":
    index_resume()

    while True:
        question = input("\nAsk about your resume (or 'exit'): ")

        if question.lower() == "exit":
            break

        for i, chunk in enumerate(
            search_resume(question), start=1
        ):
            print(f"\n--- Result {i} ---")
            print(chunk)
