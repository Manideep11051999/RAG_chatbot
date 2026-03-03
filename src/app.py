"""
Streamlit Chat Interface for the RAG Chatbot.

This is a starter template — feel free to modify, extend, or replace it entirely.
Run with: streamlit run src/app.py
"""

import os
import streamlit as st

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


# ──────────────────────────────────────────────
# UI Components
# ──────────────────────────────────────────────

def render_sidebar() -> dict:
    """Render sidebar settings. Returns a dict of user-configured parameters."""
    with st.sidebar:
        st.header("⚙️ Settings")

        top_k = st.slider(
            "Retrieved chunks (Top-K)",
            min_value=1,
            max_value=10,
            value=3,
            help="How many document chunks to retrieve per query.",
        )

        st.divider()
        st.markdown("**How it works**")
        st.markdown(
            "1. Your question is embedded\n"
            "2. Relevant document chunks are retrieved\n"
            "3. An LLM generates an answer based on the context"
        )

    return {"top_k": top_k}


def render_message(message: dict) -> None:
    """Render a single chat message with optional source expander."""
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📄 Sources"):
                for source in message["sources"]:
                    st.markdown(f"- {source}")


def render_chat_history() -> None:
    """Display all messages stored in session state."""
    for message in st.session_state.messages:
        render_message(message)


@st.cache_resource(show_spinner="📚 Building / loading knowledge base (first run can take a bit)...")
def get_vector_store() -> Chroma:
    """
    Load documents -> chunk -> embed -> store in Chroma.
    Cached so it runs once per session (fast afterwards).
    """
    data_path = os.path.join("data", "knowledge_base")
    persist_dir = "chroma_db"

    # Embeddings model (Ollama)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # If already built, load from disk
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    # Otherwise: load docs from the knowledge base
    loader = DirectoryLoader(
        data_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True,
    )
    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    # Create + persist vector DB
    vector_store = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    vector_store.persist()
    return vector_store


def build_prompt(context: str, question: str) -> list:
    """
    Create a strict RAG prompt: answer only from context + cite sources.
    """
    template = """
You are a helpful customer service assistant for a large automotive company.

RULES:
- Use ONLY the information in the provided CONTEXT.
- If the answer is not in the context, say: "I don't know based on the provided documents."
- Keep the answer clear and concise.


CONTEXT:
{context}

QUESTION:
{question}
"""
    prompt = ChatPromptTemplate.from_template(template)
    return prompt.format_messages(context=context, question=question)

def get_bot_response(query: str, top_k: int) -> tuple[str, list[str]]:
    # 1) Retrieve relevant chunks
    vector_store = get_vector_store()
    docs = vector_store.similarity_search(query, k=top_k)

    # Build context + sources
    context_parts = []
    sources = []
    for d in docs:
        context_parts.append(d.page_content)
        # DirectoryLoader/TextLoader stores file path in metadata["source"]
        src = d.metadata.get("source", "Unknown source")
        s = os.path.basename(src)
        if s not in sources:
            sources.append(s)

    context = "\n\n---\n\n".join(context_parts)

    # 2) Call the LLM (Ollama chat model)
    llm = ChatOllama(model="llama3.2:3b", temperature=0)

    messages = build_prompt(context=context, question=query)
    resp = llm.invoke(messages)

    # 3) Return answer + sources (sources also shown in UI expander)
    answer = resp.content.strip()

    # Add sources line (the UI already shows sources, but this helps the requirement)
    answer_with_sources = answer + "\n\n**Sources:**\n" + "\n".join([f"- {s}" for s in sources])

    return answer_with_sources, sources

# ──────────────────────────────────────────────
# Main App
# ──────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Customer Service Chatbot",
        page_icon="🚗",
        layout="centered",
    )

    st.title("🚗 Customer Service Chatbot")
    st.caption("Ask questions about vehicles, services, warranty, and more.")

    # Sidebar
    settings = render_sidebar()

    # Session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    render_chat_history()

    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        # User message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Bot response
        answer, sources = get_bot_response(prompt, top_k=settings["top_k"])

        response = {"role": "assistant", "content": answer, "sources": sources}
        render_message(response)
        st.session_state.messages.append(response)


if __name__ == "__main__":
    main()
