import os
from pathlib import Path

import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


# =========================================================
# CONFIG
# =========================================================

DATA_FILE = Path("data/support.txt")
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "teckinfo_support"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="Teckinfo AI Support Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Teckinfo AI Support Assistant")
st.caption("RAG-based Technical Support Assistant")


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


embedding_model = load_embedding_model()


# =========================================================
# LOAD LLM
# =========================================================

@st.cache_resource
def load_llm():

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL
    )

    return tokenizer, model


tokenizer, llm = load_llm()


# =========================================================
# CHUNKING
# =========================================================

def create_chunks(text):

    chunk_size = 200
    overlap = 40

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(words[start:end])

        chunks.append(chunk)

        start = end - overlap

    return chunks


# =========================================================
# CHROMADB
# =========================================================

@st.cache_resource
def load_collection():

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    try:

        collection = client.get_collection(
            name=COLLECTION_NAME
        )

        # If database already contains documents,
        # don't rebuild it.
        if collection.count() > 0:
            return collection

    except Exception:

        pass


    # -----------------------------------------------------
    # Build database automatically
    # -----------------------------------------------------

    if not DATA_FILE.exists():

        st.error(
            "data/support.txt not found."
        )

        st.stop()


    with open(
        DATA_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        text = f.read()


    chunks = create_chunks(text)


    st.info(
        "Building knowledge base for the first time..."
    )


    embeddings = embedding_model.encode(
        chunks,
        show_progress_bar=False
    ).tolist()


    # Delete old collection if partially created
    try:

        client.delete_collection(
            name=COLLECTION_NAME
        )

    except Exception:

        pass


    collection = client.create_collection(
        name=COLLECTION_NAME
    )


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
# RAG FUNCTION
# =========================================================

def generate_answer(query):

    # -----------------------------------------------------
    # Embedding
    # -----------------------------------------------------

    query_embedding = embedding_model.encode(
        query
    ).tolist()


    # -----------------------------------------------------
    # Retrieval
    # -----------------------------------------------------

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1
    )


    context = results["documents"][0][0]


    # -----------------------------------------------------
    # Prompt
    # -----------------------------------------------------

    messages = [

        {
            "role": "system",

            "content": """
You are a technical support assistant for Teckinfo.

Answer the user's question using ONLY the provided context.

Do not use outside knowledge.
Do not guess.

If the answer is not available in the context, say:

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

    ]


    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True
    )


    outputs = llm.generate(

        **inputs,

        max_new_tokens=120,

        do_sample=False,

        pad_token_id=tokenizer.eos_token_id
    )


    input_length = inputs["input_ids"].shape[1]


    generated_tokens = outputs[
        0,
        input_length:
    ]


    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()


    return answer


# =========================================================
# CHAT HISTORY
# =========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )


# =========================================================
# CHAT INPUT
# =========================================================

query = st.chat_input(
    "Ask your Teckinfo support question..."
)


if query:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )


    with st.chat_message("user"):

        st.markdown(query)


    with st.chat_message("assistant"):

        with st.spinner("Searching knowledge base..."):

            answer = generate_answer(query)


        st.markdown(answer)


    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("Project Information")

    st.write(
        "### Architecture"
    )

    st.write(
        "User Question → Embedding → ChromaDB → "
        "Retrieved Context → Qwen LLM → Answer"
    )

    st.write(
        "### Models"
    )

    st.write(
        f"Embedding: `{EMBEDDING_MODEL}`"
    )

    st.write(
        f"LLM: `{LLM_MODEL}`"
    )

    st.write(
        "Vector DB: `ChromaDB`"
    )

    st.write(
        "Chunk Size: `200 words`"
    )

    st.write(
        "Overlap: `40 words`"
    )


    if st.button("Clear Chat"):

        st.session_state.messages = []

        st.rerun()