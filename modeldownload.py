from sentence_transformers import SentenceTransformer

# Downloads and loads the model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Saves it to your specified path
model.save('/home/drut/chatbot/models/all-mpnet-base-v2')
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="/home/drut/chatbot/models/all-mpnet-base-v2")
embedding = embeddings.embed_query("This is a test sentence.")
print(len(embedding))  # Should be 768
from sentence_transformers import SentenceTransformer

# Load and download the model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Save it locally (change the path as needed)
model.save("/home/drut/chatbot/models/miniLM")
print(model.encode("This is a test sentence").shape)  # Output: (384,)

