import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="Teckinfo AI Support Assistant",
    page_icon="🤖",
    layout="centered"
)


# =========================================
# HEADER
# =========================================

st.title("🤖 Teckinfo AI Support Assistant")
st.caption("AI-powered technical support using RAG")


# =========================================
# LOAD MODELS
# =========================================

@st.cache_resource
def load_models():

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct"
    )

    llm = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct"
    )

    return embedding_model, tokenizer, llm


with st.spinner("Loading AI models..."):

    embedding_model, tokenizer, llm = load_models()


# =========================================
# LOAD CHROMADB
# =========================================

@st.cache_resource
def load_database():

    client = chromadb.PersistentClient(
        path="chroma_db"
    )

    collection = client.get_collection(
        name="teckinfo_support"
    )

    return collection


collection = load_database()


# =========================================
# CHAT HISTORY
# =========================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# =========================================
# DISPLAY CHAT HISTORY
# =========================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# =========================================
# USER INPUT
# =========================================

query = st.chat_input(
    "Ask your Teckinfo support question..."
)


# =========================================
# PROCESS QUESTION
# =========================================

if query:

    # -------------------------------
    # Display user message
    # -------------------------------

    with st.chat_message("user"):

        st.markdown(query)

    st.session_state.messages.append({
        "role": "user",
        "content": query
    })


    # -------------------------------
    # Create embedding
    # -------------------------------

    query_embedding = embedding_model.encode(
        query
    ).tolist()


    # -------------------------------
    # Retrieve context
    # -------------------------------

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1
    )

    context = results["documents"][0][0]


    # -------------------------------
    # Create Qwen messages
    # -------------------------------

    messages = [
        {
            "role": "system",
            "content": """You are a technical support assistant for Teckinfo.

Answer the user's question using ONLY the provided context.

Do not use outside knowledge.
Do not guess.

If the answer is not available in the context, say:
Information not available in the provided documentation."""
        },
        {
            "role": "user",
            "content": f"""Context:

{context}

Question:

{query}"""
        }
    ]


    # -------------------------------
    # Apply Qwen chat template
    # -------------------------------

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


    # -------------------------------
    # Tokenize
    # -------------------------------

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True
    )


    # -------------------------------
    # Generate answer
    # -------------------------------

    with st.spinner("Thinking..."):

        outputs = llm.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )


    # -------------------------------
    # Get generated tokens only
    # -------------------------------

    input_length = inputs["input_ids"].shape[1]

    generated_tokens = outputs[
        0,
        input_length:
    ]


    # -------------------------------
    # Decode
    # -------------------------------

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()


    # -------------------------------
    # Display answer
    # -------------------------------

    with st.chat_message("assistant"):

        st.markdown(answer)


    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })


# =========================================
# SIDEBAR
# =========================================

with st.sidebar:

    st.header("⚙️ Settings")

    st.write("**Model:** Qwen2.5-1.5B-Instruct")

    st.write("**Embedding:** all-MiniLM-L6-v2")

    st.write("**Vector Database:** ChromaDB")

    st.write("**Architecture:** RAG")

    st.divider()

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []

        st.rerun()