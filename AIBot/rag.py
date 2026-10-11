
from pathlib import Path

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb


# Paths
BASE_DIR = Path(__file__).resolve().parent
RESUME_PATH = BASE_DIR / "resume.pdf"
DB_PATH = BASE_DIR / "chroma_db"

# Load embedding model
print("Loading embedding model...")
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector database
chroma_client = chromadb.PersistentClient(path=str(DB_PATH))

collection = chroma_client.get_or_create_collection(
    name="resume",
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
        chunk_words = words[start:start + chunk_size]

        if not chunk_words:
            break

        chunks.append(" ".join(chunk_words))

        if start + chunk_size >= len(words):
            break

    return chunks


def index_resume():
    if not RESUME_PATH.exists():
        raise FileNotFoundError(
            f"Resume not found: {RESUME_PATH}"
        )

    text = extract_text(RESUME_PATH)

    if not text.strip():
        raise ValueError(
            "No text extracted. The PDF may be scanned."
        )

    chunks = split_into_chunks(text)

    # Replace previous indexed chunks
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    embeddings = embedding_model.encode(
        chunks,
        normalize_embeddings=True
    ).tolist()

    collection.add(
        ids=[f"chunk-{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings
    )

    print(f"Indexed {len(chunks)} chunks.")


def search_resume(question, top_k=3):
    if collection.count() == 0:
        raise RuntimeError(
            "Resume is not indexed. Run index_resume() first."
        )

    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count())
    )

    return results["documents"][0]


if __name__ == "__main__":
    index_resume()

    while True:
        question = input("\nAsk about your resume (or 'exit'): ")

        if question.lower() == "exit":
            break

        chunks = search_resume(question)

        print("\nRelevant resume sections:")
        for i, chunk in enumerate(chunks, start=1):
            print(f"\n--- Result {i} ---")
            print(chunk)
