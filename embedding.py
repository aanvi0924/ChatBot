import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define relative paths
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
MODEL_DIR = os.path.join(SCRIPT_DIR, "models", "miniLM")
CHUNKS_DIR = os.path.join(DATA_DIR, "chunks")
PDF_PATH = os.path.join(DATA_DIR, "DSP_Installation_Guide.pdf")
JSON_PATH = os.path.join(DATA_DIR, "dsp_elements.json")
CSV_PATH = os.path.join(DATA_DIR, "dsp_elements.csv")
CHROMA_DB_DIR = os.path.join(DATA_DIR, "chromadb")
import requests

url = 'https://storage.googleapis.com/workbench_datasets/ChatBot/DSP%20Installation%20Guide.pdf'
os.makedirs(DATA_DIR, exist_ok=True)

response = requests.get(url)
with open(PDF_PATH, 'wb') as f:
    f.write(response.content)

print(f"Downloaded to {PDF_PATH}")
from langchain_community.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader(PDF_PATH)
data = loader.load()

from langchain_core.documents import Document
import json

json_data = [doc.dict() for doc in data]
with open(JSON_PATH, 'w') as f:
    json.dump(json_data, f, indent=2)

print(f"Saved to: {JSON_PATH}")
import pandas as pd

with open(JSON_PATH, 'r') as f:
    docs = json.load(f)

rows = []
for item in docs:
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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

combined_text = " ".join(df['page_content'].dropna().astype(str).tolist())
document = Document(page_content=combined_text)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents([document])

os.makedirs(CHUNKS_DIR, exist_ok=True)

for i, chunk in enumerate(chunks):
    with open(os.path.join(CHUNKS_DIR, f"chunk_{i+1}.txt"), "w", encoding="utf-8") as f:
        f.write(chunk.page_content)

print(f"Saved {len(chunks)} chunks in {CHUNKS_DIR}")
import chromadb
from sentence_transformers import SentenceTransformer

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
query = "How is DSP Orchestration started?"
query_embedding = embedder.encode([query]).tolist()

results = collection.query(query_embeddings=query_embedding, n_results=3)
print("Top matches:\n", results["documents"][0])
