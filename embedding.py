import requests
import json
import pandas as pd
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_core.documents import Document

# Step 1: Download the PDF
url = 'https://storage.googleapis.com/workbench_datasets/ChatBot/DSP%20Installation%20Guide.pdf'
local_path = '/home/drut/chatbot/data/DSP_Installation_Guide.pdf'

response = requests.get(url)
with open(local_path, 'wb') as f:
    f.write(response.content)

print(f"Downloaded to {local_path}")

# Step 2: Load PDF using LangChain's UnstructuredPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader("/home/drut/chatbot/data/DSP_Installation_Guide.pdf")
data = loader.load()




# Load the document
data = loader.load()

# Step 3: Convert LangChain documents to serializable dicts and save as JSON
output_path_json = '/home/drut/chatbot/data/dsp_elements.json'

# Convert documents to dicts
json_data = [doc.dict() for doc in data]

# Save to JSON
with open(output_path_json, 'w') as f:
    json.dump(json_data, f, indent=2)

print(f"Saved to: {output_path_json}")

# Step 4: Load JSON for further processing
with open(output_path_json, 'r') as f:
    docs = json.load(f)

# Extract page content for embedding
texts = [item["page_content"] for item in docs]

# Step 5: Convert to LangChain Documents
documents = [
    Document(page_content=item["page_content"], metadata=item["metadata"])
    for item in docs
]

# Step 6: Flatten relevant fields and save to CSV
rows = []
for item in docs:
    row = {
        "page_content": item.get("page_content", ""),
        "page_number": item.get("metadata", {}).get("page_number", ""),
        "category": item.get("metadata", {}).get("category", ""),
        "filename": item.get("metadata", {}).get("filename", "")
    }
    rows.append(row)

# Convert to DataFrame
df = pd.DataFrame(rows)

# Save DataFrame to CSV
output_path_csv = '/home/drut/chatbot/data/dsp_elements.csv'
df.to_csv(output_path_csv, index=False)

print(f"CSV saved to: {output_path_csv}")

# Step 7: Load the CSV, clean, and convert to LangChain Documents
df = pd.read_csv(output_path_csv)

# Drop rows where 'page_content' is NaN or empty
df = df[df['page_content'].notna() & df['page_content'].str.strip().astype(bool)]

# Convert to LangChain Documents
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
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="/home/drut/chatbot/models/all-mpnet-base-v2")

import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import os
# Step 2: Load CSV
df = pd.read_csv("/home/drut/chatbot/data/dsp_elements.csv")

# Step 3: Combine all text from the 'page_content' column
combined_text = " ".join(df['page_content'].dropna().astype(str).tolist())

# Step 4: Create a Document and split into chunks
document = Document(page_content=combined_text)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents([document])

# Step 5: Save chunks into the drut/chunks folder
output_dir = "/home/drut/chatbot/data/chunks"
os.makedirs(output_dir, exist_ok=True)

for i, chunk in enumerate(chunks):
    with open(f"{output_dir}/chunk_{i+1}.txt", "w", encoding="utf-8") as f:
        f.write(chunk.page_content)

print(f" Saved {len(chunks)} chunks in {output_dir}")
print("Available columns:", df.columns.tolist())
import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# 1. Load embedding model
embedder = SentenceTransformer("/home/drut/chatbot/models/miniLM")

# 2. Read all .txt files from 'chunk' folder
chunk_dir = "/home/drut/chatbot/data/chunks"
documents = []
ids = []

for i, filename in enumerate(os.listdir(chunk_dir)):
    if filename.endswith(".txt"):
        with open(os.path.join(chunk_dir, filename), 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content:  # Skip empty files
                documents.append(content)
                ids.append(f"chunk_{i}")

# 3. Generate embeddings
embeddings = embedder.encode(documents).tolist()

# 4. Initialize persistent ChromaDB client
client = chromadb.PersistentClient(path="/home/drut/chatbot/data/chromadb")

# 5. Create or get collection
collection = client.get_or_create_collection(name="dsp_docs")

# 6. Insert into collection
collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=ids,
    metadatas=[{"source": "chunk_txt"} for _ in documents]
)

print(f" Inserted {len(documents)} chunked text files into ChromaDB.")
query = "How is DSP Orchestration started?"
query_embedding = embedder.encode([query]).tolist()

results = collection.query(query_embeddings=query_embedding, n_results=3)

print("Top matches:\n", results["documents"][0])


