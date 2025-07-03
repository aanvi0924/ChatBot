import os
import json
import torch
import spacy
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util

# === Load models ===
embedding_model_path = os.path.join("models", "miniLM")
embedding_model = SentenceTransformer(embedding_model_path)

# Local mistral model (adjust to your path)
model_path = os.path.join("finetuningllm", "mistral-7b-instruct")
tokenizer = AutoTokenizer.from_pretrained(model_path)
llm_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

# === Load spaCy model from local directory ===
import spacy
nlp = spacy.load("en_core_web_sm")


# === Data file ===
data_file = os.path.join("ragas_test_data.json")
with open(data_file, "r") as f:
    dataset = json.load(f)

# === Metric functions ===
def answer_similarity(ans1, ans2):
    return util.cos_sim(embedding_model.encode(ans1), embedding_model.encode(ans2)).item()

def answer_relevancy(question, answer):
    return util.cos_sim(embedding_model.encode(question), embedding_model.encode(answer)).item()

def compute_precision_recall(gt, retrieved):
    gt_tokens = set(gt.lower().split())
    ret_tokens = set(retrieved.lower().split())
    common = gt_tokens & ret_tokens
    precision = len(common) / len(ret_tokens) if ret_tokens else 0
    recall = len(common) / len(gt_tokens) if gt_tokens else 0
    return precision, recall

def context_entity_recall(gt, retrieved):
    gt_ents = {ent.text for ent in nlp(gt).ents}
    ret_ents = {ent.text for ent in nlp(retrieved).ents}
    if not gt_ents:
        return 1.0
    return len(gt_ents & ret_ents) / len(gt_ents)

def is_faithful(context, answer):
    prompt = f"""
    [INST] Is the following answer faithful to the provided context? Answer "yes" or "no".
    
    Context: {context}
    
    Answer: {answer}
    [/INST]
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(llm_model.device)
    with torch.no_grad():
        output = llm_model.generate(**inputs, max_new_tokens=30)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True).lower()
    return 1.0 if "yes" in decoded else 0.0

# === Evaluate ===
results = []
for item in dataset:
    question = item["question"]
    ground_truth = item["ground_truth"]
    generated_answer = item["answer"]
    contexts = " ".join(item["contexts"])

    sim = answer_similarity(generated_answer, ground_truth)
    rel = answer_relevancy(question, generated_answer)
    prec, rec = compute_precision_recall(ground_truth, contexts)
    ent_rec = context_entity_recall(ground_truth, contexts)
    faith = is_faithful(contexts, generated_answer)

    results.append({
        "question": question,
        "answer_similarity": sim,
        "answer_relevancy": rel,
        "context_precision": prec,
        "context_recall": rec,
        "context_entity_recall": ent_rec,
        "faithfulness": faith
    })

# === Save output ===
out_path = os.path.join("outputs", "offline_metrics.json")
os.makedirs("outputs", exist_ok=True)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(" Evaluation complete. Results saved to:", out_path)
import pandas as pd

# Convert to DataFrame
df = pd.DataFrame(results)

# Calculate average of each metric
summary = df[[
    "answer_similarity",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "context_entity_recall",
    "faithfulness"
]].mean()

print("\n Model Evaluation Summary:\n")
print(summary.round(3))

