import chromadb
from sentence_transformers import SentenceTransformer


# =========================================
# 1. Load Embedding Model
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
# 3. Ask Question
# =========================================

query = input("Ask your question: ")


# =========================================
# 4. Convert Question into Embedding
# =========================================

query_embedding = embedding_model.encode(
    query
).tolist()


# =========================================
# 5. Search Top 10 Relevant Chunks
# =========================================

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=10,
    include=["documents", "distances"]
)


# =========================================
# 6. Display Results
# =========================================

print("\n========== RETRIEVED INFORMATION ==========\n")

for i, (document, distance) in enumerate(
    zip(
        results["documents"][0],
        results["distances"][0]
    )
):

    print(f"--- Result {i + 1} ---")
    print(f"Distance: {distance:.4f}")
    print(document)
    print()


print("============================================")