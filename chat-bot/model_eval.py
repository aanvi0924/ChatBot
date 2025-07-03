import os
import json
import torch
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    answer_similarity,
    context_precision,
    context_recall,
    context_entity_recall,
)
from ragas.llms import TransformersLLM, set_llm
from ragas.embeddings import SentenceTransformersEmbeddings, set_embeddings
from transformers import AutoTokenizer, AutoModelForCausalLM

# === Step 1: Load dataset from relative path ===
current_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(current_dir, "ragas_test_data.json")

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

dataset = Dataset.from_list(data)

# === Step 2: Load Mistral model from local path ===
mistral_path = os.path.join(current_dir, "finetuningllm", "mistral-7b-instruct")
device = "cuda" if torch.cuda.is_available() else "cpu"

llm = TransformersLLM(
    model_id=mistral_path,
    model_kwargs={
        "device_map": "auto",
        "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
    },
    tokenizer_id=mistral_path
)

set_llm(llm)

# === Step 3: Load MiniLM embeddings from local path ===
minilm_path = os.path.join(current_dir, "models", "miniLM")
embedding_model = SentenceTransformersEmbeddings(minilm_path)
set_embeddings(embedding_model)

# === Step 4: Define metrics ===
metrics = [
    faithfulness,
    answer_relevancy,
    answer_similarity,
    context_precision,
    context_recall,
    context_entity_recall,
]

# === Step 5: Evaluate ===
results = evaluate(dataset, metrics)

# === Step 6: Print results ===
print("\nRAGAS Evaluation Results:")
for metric, score in results.items():
    print(f"{metric}: {score:.4f}")
