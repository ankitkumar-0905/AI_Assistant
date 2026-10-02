import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


# =========================================
# 1. Embedding Model
# =========================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# =========================================
# 2. Connect to ChromaDB
# =========================================

client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = client.get_collection(
    name="teckinfo_support"
)


# =========================================
# 3. Load Qwen LLM
# =========================================

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct"
)

llm = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct"
)


# =========================================
# 4. Chat
# =========================================

print("\n==========================================")
print("      TECKINFO AI SUPPORT ASSISTANT")
print("==========================================")
print("Type 'exit' to stop.\n")


while True:

    query = input("You: ")

    if query.lower() == "exit":
        print("\nChatbot closed.")
        break


    # =====================================
    # 5. Create Query Embedding
    # =====================================

    query_embedding = embedding_model.encode(
        query
    ).tolist()


    # =====================================
    # 6. Retrieve Best Context
    # =====================================

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1
    )

    context = results["documents"][0][0]


    # =====================================
    # 7. Show Retrieved Context
    # =====================================

    print("\n[Retrieved context found]")


    # =====================================
    # 8. Create Chat Messages
    # =====================================

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


    # =====================================
    # 9. Apply Qwen Chat Template
    # =====================================

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


    # =====================================
    # 10. Tokenize
    # =====================================

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True
    )


    # =====================================
    # 11. Generate
    # =====================================

    outputs = llm.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )


    # =====================================
    # 12. Get Only Newly Generated Tokens
    # =====================================

    input_length = inputs["input_ids"].shape[1]

    generated_tokens = outputs[
        0,
        input_length:
    ]


    # =====================================
    # 13. Decode Answer
    # =====================================

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()


    # =====================================
    # 14. Display
    # =====================================

    print("\nAI:", answer)
    print()