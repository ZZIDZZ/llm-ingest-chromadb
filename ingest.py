import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from utils.extract import extract_text

# 1. Initialize ChromaDB client
client = chromadb.Client(
    Settings(
        persist_directory="./chroma_db"  # store DB contents here
    )
)

collection = client.get_or_create_collection("my_documents")

# 2. Load Sentence Transformers model
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(text: str):
    """Convert a string to vector embedding."""
    return model.encode(text).tolist()  # .tolist() converts NumPy array to Python list

def ingest_documents(folder_path: str):
    """Loop through documents, extract text, and store in ChromaDB."""
    # Supported file extensions
    supported_extensions = {"pdf", "docx", "xls", "xlsx", "pptx"}

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        if not os.path.isfile(file_path):
            # Skip directories or non-files
            continue

        # Check extension
        extension = file_name.lower().split(".")[-1]
        if extension not in supported_extensions:
            print(f"Skipping {file_name} (unsupported file type: .{extension})")
            continue

        # Extract text
        text = extract_text(file_path)
        if text and text.strip():
            doc_id = file_name
            embedding = embed_text(text)

            collection.add(
                documents=[text],  # The original text
                embeddings=[embedding],
                ids=[doc_id],
                metadatas=[{"filename": file_name}],
            )
            print(f"Ingested: {file_name}")
        else:
            print(f"No text extracted from: {file_name}")

if __name__ == "__main__":
    folder_path = "./docs-data"  # Put all your docs here
    ingest_documents(folder_path)
