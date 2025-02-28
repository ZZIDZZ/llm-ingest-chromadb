# chatbot_app.py
import gradio as gr
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Initialize Chroma client & collection
client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="./chroma_db"))
collection = client.get_collection("my_documents")

# Load model for query embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_relevant_docs(query, top_k=3):
    """Search ChromaDB for similar documents."""
    query_embedding = model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results

def chatbot_interface(user_query):
    # 1. Retrieve relevant docs
    results = get_relevant_docs(user_query)

    # 2. Format retrieval data
    docs_found = []
    for doc_id, metadata, document in zip(results["ids"][0], results["metadatas"][0], results["documents"][0]):
        # doc_id = an ID, metadata = e.g. {"filename": "sample.pdf"}, document = text chunk
        docs_found.append(f"**Document ID**: {doc_id}\n**Filename**: {metadata.get('filename')}\n\n{document[:500]}...")

    # If you have a separate LLM step, you can do something like:
    # combined_context = "\n\n".join(results["documents"][0])
    # answer = run_llm_with_context(user_query, combined_context)
    # But here's a simple example that just returns the docs:

    if len(docs_found) == 0:
        return "No relevant documents found."
    else:
        return "\n\n---\n\n".join(docs_found)

# Gradio UI
iface = gr.Interface(
    fn=chatbot_interface,
    inputs="text",
    outputs="markdown",
    title="Document Retrieval Chatbot",
    description="Ask questions about your PDF, DOCX, Excel, PPT documents."
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
