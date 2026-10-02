import os
from pathlib import Path

import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq


# =========================================================
# CONFIGURATION
# =========================================================

DATA_FILE = Path("data/support.txt")
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "teckinfo_support"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"


# =========================================================
# STREAMLIT PAGE
# =========================================================

st.set_page_config(
    page_title="Teckinfo AI Support Assistant",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("🤖 Teckinfo AI Support Assistant")

st.caption(
    "RAG-based Technical Support Assistant using "
    "ChromaDB + Embeddings + Groq LLM"
)


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(EMBEDDING_MODEL)


embedding_model = load_embedding_model()


# =========================================================
# GROQ CLIENT
# =========================================================

@st.cache_resource
def load_groq_client():

    api_key = st.secrets.get("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not configured in Streamlit Secrets."
        )

    return Groq(api_key=api_key)


groq_client = load_groq_client()


# =========================================================
# CHUNKING
# =========================================================

def create_chunks(text, chunk_size=200, overlap=40):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# =========================================================
# LOAD / CREATE CHROMA DATABASE
# =========================================================

@st.cache_resource
def load_collection():

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    # Check whether collection already exists
    try:

        collection = client.get_collection(
            name=COLLECTION_NAME
        )

        count = collection.count()

        if count > 0:
            return collection

    except Exception:
        pass


    # -----------------------------------------------------
    # If collection does not exist, create it
    # -----------------------------------------------------

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"Support documentation not found: {DATA_FILE}"
        )


    with open(
        DATA_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        text = file.read()


    # Create chunks
    chunks = create_chunks(
        text,
        chunk_size=200,
        overlap=40
    )


    # Generate embeddings
    embeddings = embedding_model.encode(
        chunks,
        show_progress_bar=False
    ).tolist()


    # Delete old collection if present
    try:

        client.delete_collection(
            name=COLLECTION_NAME
        )

    except Exception:
        pass


    # Create new collection
    collection = client.create_collection(
        name=COLLECTION_NAME
    )


    # Add documents
    ids = [
        f"chunk_{i}"
        for i in range(len(chunks))
    ]


    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings
    )


    return collection


collection = load_collection()


# =========================================================
# RAG ANSWER GENERATION
# =========================================================

def generate_answer(query):

    # -----------------------------------------------------
    # STEP 1: Create query embedding
    # -----------------------------------------------------

    query_embedding = embedding_model.encode(
        query
    ).tolist()


    # -----------------------------------------------------
    # STEP 2: Retrieve relevant context
    # -----------------------------------------------------

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1
    )


    if not results["documents"]:
        return (
            "Information not available in the "
            "provided documentation."
        )


    context = results["documents"][0][0]


    # -----------------------------------------------------
    # STEP 3: Send context + question to Groq
    # -----------------------------------------------------

    response = groq_client.chat.completions.create(

        model=GROQ_MODEL,

        messages=[

            {
                "role": "system",
                "content": """
You are a technical support assistant for Teckinfo.

Answer the user's question using ONLY the provided context.

Rules:
1. Do not use outside knowledge.
2. Do not guess.
3. Give clear and practical technical answers.
4. If the answer is not available in the context, say exactly:

Information not available in the provided documentation.
"""
            },

            {
                "role": "user",
                "content": f"""
Context:

{context}

Question:

{query}
"""
            }

        ],

        temperature=0,

        max_tokens=150
    )


    # -----------------------------------------------------
    # STEP 4: Extract answer
    # -----------------------------------------------------

    answer = response.choices[0].message.content

    return answer.strip()


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ System Information")

    st.write(
        "**Architecture:** RAG"
    )

    st.write(
        "**Embedding:** all-MiniLM-L6-v2"
    )

    st.write(
        "**Vector Database:** ChromaDB"
    )

    st.write(
        "**LLM:** Groq"
    )

    st.write(
        f"**Model:** {GROQ_MODEL}"
    )

    st.write(
        "**Knowledge Base:** support.txt"
    )


    st.divider()


    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# =========================================================
# USER INPUT
# =========================================================

query = st.chat_input(
    "Ask your Teckinfo support question..."
)


if query:

    # -----------------------------------------------------
    # Display user message
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )


    with st.chat_message("user"):

        st.markdown(query)


    # -----------------------------------------------------
    # Generate AI answer
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching documentation and generating answer..."
        ):

            try:

                answer = generate_answer(
                    query
                )

            except Exception as e:

                answer = (
                    "⚠️ Error while generating answer:\n\n"
                    f"{str(e)}"
                )


        st.markdown(answer)


    # -----------------------------------------------------
    # Save assistant response
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )