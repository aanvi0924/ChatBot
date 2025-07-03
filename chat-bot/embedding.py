import os
from dotenv import load_dotenv
import requests
import json
import pandas as pd
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# === Load environment variable ===
load_dotenv("embedding.env")
url = os.getenv("PDF_URL")
if not url:
    raise ValueError("PDF_URL environment variable is not set.")

# === Setup directories ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
MODEL_DIR = os.path.join(SCRIPT_DIR, "models", "miniLM")
CHUNKS_DIR = os.path.join(DATA_DIR, "chunks")
PDF_PATH = os.path.join(DATA_DIR, "DSP_Installation_Guide.pdf")
JSON_PATH = os.path.join(DATA_DIR, "dsp_elements.json")
CSV_PATH = os.path.join(DATA_DIR, "dsp_elements.csv")
CHROMA_DB_DIR = os.path.join(DATA_DIR, "chromadb")

os.makedirs(DATA_DIR, exist_ok=True)

# === Download PDF ===
response = requests.get(url)
with open(PDF_PATH, 'wb') as f:
    f.write(response.content)
print(f"Downloaded to {PDF_PATH}")

# === Load PDF and save JSON ===
loader = PyMuPDFLoader(PDF_PATH)
data = loader.load()
json_data = [doc.dict() for doc in data]

with open(JSON_PATH, 'w') as f:
    json.dump(json_data, f, indent=2)
print(f"Saved to: {JSON_PATH}")

# === Convert to CSV ===
rows = []
for item in json_data:
    row = {
        "page_content": item.get("page_content", ""),
        "page_number": item.get("metadata", {}).get("page_number", ""),
        "category": item.get("metadata", {}).get("category", ""),
        "filename": item.get("metadata", {}).get("filename", "")
    }
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(CSV_PATH, index=False)
print(f"CSV saved to: {CSV_PATH}")

# === Filter and create LangChain documents ===
df = pd.read_csv(CSV_PATH)
df = df[df['page_content'].notna() & df['page_content'].str.strip().astype(bool)]

documents = [
    Document(
        page_content=row["page_content"],
        metadata={
            "page_number": row.get("page_number", ""),
            "category": row.get("category", "")
        }
    )
    for _, row in df.iterrows()
]
print(f"Documents ready: {len(documents)}")

# === Chunking ===
combined_text = " ".join(df['page_content'].dropna().astype(str).tolist())
document = Document(page_content=combined_text)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents([document])

os.makedirs(CHUNKS_DIR, exist_ok=True)
for i, chunk in enumerate(chunks):
    with open(os.path.join(CHUNKS_DIR, f"chunk_{i+1}.txt"), "w", encoding="utf-8") as f:
        f.write(chunk.page_content)
print(f"Saved {len(chunks)} chunks in {CHUNKS_DIR}")

# === Embedding and Storing in ChromaDB ===
embedder = SentenceTransformer(MODEL_DIR)
documents = []
ids = []

for i, filename in enumerate(os.listdir(CHUNKS_DIR)):
    if filename.endswith(".txt"):
        with open(os.path.join(CHUNKS_DIR, filename), 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content:
                documents.append(content)
                ids.append(f"chunk_{i}")

embeddings = embedder.encode(documents).tolist()
client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
collection = client.get_or_create_collection(name="dsp_docs")

collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=ids,
    metadatas=[{"source": "chunk_txt"} for _ in documents]
)
print(f"Inserted {len(documents)} chunked text files into ChromaDB.")

# === Query Example ===
query = "How is DSP Orchestration started?"
query_embedding = embedder.encode([query]).tolist()
results = collection.query(query_embeddings=query_embedding, n_results=3)
print("Top matches:\n", results["documents"][0])
print("Top match IDs:\n", results["ids"][0])
print("Top match scores:\n", results["distances"][0])