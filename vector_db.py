from pathlib import Path

from sentence_transformers import SentenceTransformer
import chromadb


# =========================================
# 1. Read support.txt
# =========================================

file_path = Path("data/support.txt")

with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()


# =========================================
# 2. Chunk Settings
# =========================================

chunk_size = 200
overlap = 40


# =========================================
# 3. Create Chunks
# =========================================

words = text.split()

chunks = []

start = 0

while start < len(words):

    end = start + chunk_size

    chunk = " ".join(words[start:end])

    chunks.append(chunk)

    start = end - overlap


print("Total words:", len(words))
print("Total chunks:", len(chunks))


# =========================================
# 4. Load Embedding Model
# =========================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# =========================================
# 5. Create ChromaDB
# =========================================

client = chromadb.PersistentClient(
    path="chroma_db"
)


# Delete old collection if it exists
try:
    client.delete_collection(
        name="teckinfo_support"
    )
    print("Old collection deleted.")

except Exception:
    pass


# Create new collection
collection = client.create_collection(
    name="teckinfo_support"
)


# =========================================
# 6. Create Embeddings
# =========================================

embeddings = embedding_model.encode(
    chunks,
    show_progress_bar=True
).tolist()


# =========================================
# 7. Create IDs
# =========================================

ids = [
    f"chunk_{i}"
    for i in range(len(chunks))
]


# =========================================
# 8. Store in ChromaDB
# =========================================

collection.add(
    ids=ids,
    documents=chunks,
    embeddings=embeddings
)


# =========================================
# 9. Done
# =========================================

print("\nEmbeddings created successfully!")
print("Documents stored in ChromaDB:", len(chunks))