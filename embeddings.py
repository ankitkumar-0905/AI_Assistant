from sentence_transformers import SentenceTransformer

# Load support document
with open("data/support.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert complete document into embedding
embedding = model.encode(text)

print("Document loaded successfully!")
print("Text length:", len(text))
print("Embedding shape:", embedding.shape)