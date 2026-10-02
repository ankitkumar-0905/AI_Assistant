from pathlib import Path


# =========================================
# 1. Read support.txt
# =========================================

file_path = Path("data/support.txt")

with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()


# =========================================
# 2. Chunk Settings
# =========================================

chunk_size = 200       # words per chunk
overlap = 40           # overlapping words


# =========================================
# 3. Convert Text into Words
# =========================================

words = text.split()


# =========================================
# 4. Create Chunks
# =========================================

chunks = []

start = 0

while start < len(words):

    end = start + chunk_size

    chunk = " ".join(words[start:end])

    chunks.append(chunk)

    start = end - overlap


# =========================================
# 5. Display Information
# =========================================

print("Total words:", len(words))
print("Total chunks:", len(chunks))


# =========================================
# 6. Show First 3 Chunks
# =========================================

for i, chunk in enumerate(chunks[:3]):

    print(f"\n--- Chunk {i + 1} ---")
    print(chunk)