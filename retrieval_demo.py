import gradio as gr
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from langdetect import detect  # for language detection
import re

# -----------------------------------------------------
# 1. Initialize ChromaDB client & get the collection
# -----------------------------------------------------
client = chromadb.PersistentClient(path='./chroma_db')
collection = client.get_collection("my_documents")

# -----------------------------------------------------
# 2. SentenceTransformer for embeddings
# -----------------------------------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------------------------------
# 3. Initialize local LLaMA model (GGUF-based)
# -----------------------------------------------------
llama_model = Llama(
    model_path="storage/qwen2.5-7b-instruct-q3_k_m.gguf",  # Path to your .gguf model
    n_gpu_layers=-1,
    main_gpu=0,
    tensor_split=[1.0],
    n_threads=10,
    n_ctx=32768,   # 32k context window
    n_batch=1024,
    use_mmap=True,
    use_mlock=False,
    offload_kqv=True,
    flash_attn=True,
    verbose=True
)

def get_relevant_docs(query, top_k=3):
    """
    Embeds the user query and retrieves the top_k
    most similar text chunks from ChromaDB.
    """
    query_embedding = embed_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results

def clean_text(text: str) -> str:
    """
    Remove null bytes and other non-printable control chars.
    """
    return re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]", "", text)

def truncate_context(context: str, max_tokens: int = 16000) -> str:
    """
    Truncate the retrieved context to avoid exceeding the model's capacity.
    Must pass UTF-8 bytes to llama-cpp-python's tokenize().
    """
    context = clean_text(context)
    context_bytes = context.encode("utf-8")  # encode to bytes
    tokens = llama_model.tokenize(context_bytes, add_bos=False)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    truncated_text = llama_model.detokenize(tokens)
    return truncated_text

def build_role_based_prompt(language, user_query, context):
    """
    Build a text prompt that simulates system & user roles,
    instructing the model to respond in the detected language.
    """
    system_prompt = (
        "You are an AI assistant that summarizes or answers in the same language as the text. "
        "Please ensure your response is in that language.\n\n"
    )
    user_prompt = (
        f"The text is in '{language}'. Summarize or answer the following content "
        f"in {language}:\n\n"
        f"{context}\n\n"
        f"Question: {user_query}\n"
    )
    final_prompt = (
        f"(role: system)\n{system_prompt}"
        f"(role: user)\n{user_prompt}"
    )
    return final_prompt

def run_llama_with_context(language, user_query, context):
    """
    Calls LLaMA to generate an answer using the combined context,
    enforcing a language restriction via a role-based prompt.
    """
    prompt = build_role_based_prompt(language, user_query, context)
    
    output = llama_model(
        prompt=prompt,
        max_tokens=512,  # Adjust for longer/shorter answers as needed
        stop=["(role: system)", "(role: user)"],
        echo=False
    )
    return output["choices"][0]["text"].strip()

def get_answer(user_query):
    """
    Orchestrates the entire retrieval & generation pipeline for a single question.
    """
    # 1. Language detection
    language = detect(user_query)  # e.g. 'en', 'es', 'fr', etc.

    # 2. Retrieve relevant docs
    results = get_relevant_docs(user_query, top_k=3)
    if not results or not results["documents"]:
        return "No relevant documents found or knowledge unavailable."

    # 3. Combine retrieved docs
    combined_context = "\n\n".join(results["documents"][0])
    truncated_context = truncate_context(combined_context, max_tokens=16000)

    # 4. Call LLaMA to answer
    answer_text = run_llama_with_context(language, user_query, truncated_context)

    # 5. (Optional) add references
    references = []
    for doc_id, meta, doc_text in zip(
        results["ids"][0],
        results["metadatas"][0],
        results["documents"][0]
    ):
        filename = meta.get("filename", "N/A")
        snippet = doc_text[:200].replace("\n", " ")
        references.append(
            f"**Doc ID**: {doc_id}, **File**: {filename}, snippet: {snippet}..."
        )
    refs_combined = "\n".join(references)

    return (
        f"**Answer** (language: {language}):\n{answer_text}\n\n"
        f"---\n\n**References:**\n{refs_combined}"
    )

# -----------------------------------------------------
# 4. Create a Chat Interface
# -----------------------------------------------------
def respond(message_history, user_query):
    # 1. Append user query to the chat history
    message_history.append((user_query, None))

    # 2. Generate the answer
    answer_text = get_answer(user_query)

    # 3. Update the most recent message pair with the bot's answer
    message_history[-1] = (user_query, answer_text)

    # 4. Return both the new state AND the chatbot messages
    return message_history, message_history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    state = gr.State([])

    user_input = gr.Textbox()
    send_button = gr.Button("Send")

    send_button.click(
        fn=respond,
        inputs=[state, user_input],
        outputs=[state, chatbot]
    )

    # This last step ensures the chatbot UI updates automatically
    # after `respond` returns (which updates the conversation).
    # We map the `state` (list of messages) onto the `chatbot` display.

    def update_chatbot(message_history):
        return message_history

    # Trigger any time the state changes to refresh the chatbot
    state.change(
        fn=update_chatbot,
        inputs=state,
        outputs=chatbot
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
