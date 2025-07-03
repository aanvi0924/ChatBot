import os
import re
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import BitsAndBytesConfig, pipeline
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

# === Define relative base directory ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# === Relative Paths ===
CHROMA_DB_DIR = os.path.join(BASE_DIR, "data", "chromadb")
EMBED_MODEL_DIR = os.path.join(BASE_DIR, "models", "miniLM")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "outputs", "checkpoint-60")
CLASSIFIER_MODEL_DIR = os.path.join(BASE_DIR, "models", "bart-large-mnli")

# === Load Embedder & Vector Store ===
embedder = SentenceTransformer(EMBED_MODEL_DIR)
client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
collection = client.get_or_create_collection(name="dsp_docs")

def retrieve_chunks(query, top_k=3):
    embedding = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=embedding, n_results=top_k)
    return results["documents"][0]

# === Load Mistral Model (Fine-tuned) ===
bnb_config = BitsAndBytesConfig(load_in_4bit=True, llm_int8_enable_fp32_cpu_offload=True)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=CHECKPOINT_DIR,
    max_seq_length=2048,
    dtype=torch.float16,
    quantization_config=bnb_config,
    device_map=0,
)
FastLanguageModel.for_inference(model)
model_device = next(model.parameters()).device

tokenizer = get_chat_template(
    tokenizer,
    chat_template="chatml",
    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
    map_eos_token=True,
)

# === Load Classifier Model from Local Path ===
classifier = pipeline(
    "zero-shot-classification",
    model=CLASSIFIER_MODEL_DIR,
    tokenizer=CLASSIFIER_MODEL_DIR
)
labels = ["small talk", "domain question"]

# === Clean up ChatML tokens ===
def clean_response(text):
    text = re.sub(r"<\|.*?\|>", "", text)  # Remove tags like <|im_start|>
    return text.strip()

# === Small Talk Generator ===
def generate_small_talk_reply(user_input):
    prompt = f"You are a helpful and friendly assistant. Reply naturally to the following message:\nUser: {user_input}\nAssistant:"
    
    messages = [{"from": "human", "value": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model_device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.8
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # === Strip unwanted ChatML tokens ===
    for token in ["<|im_start|>", "<|im_end|>", "assistant", "user"]:
        decoded = decoded.replace(token, "")
    
    decoded = decoded.strip()

    # === Strip boilerplate: take response after "Assistant:" or last line ===
    if "Assistant:" in decoded:
        return decoded.split("Assistant:")[-1].strip()
    elif "\n" in decoded:
        return decoded.split("\n")[-1].strip()

    return decoded

# === RAG Response Generator ===
def generate_response(query, context, max_tokens=300):
    prompt = f"Context: {context}\n\nQuestion: {query}"
    messages = [{"from": "human", "value": prompt}]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model_device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            use_cache=True,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # === Strip unwanted ChatML tokens if present ===
    for token in ["<|im_start|>", "<|im_end|>", "assistant", "user"]:
        decoded = decoded.replace(token, "")

    decoded = decoded.strip()

    # === Strip boilerplate: only take answer part ===
    if "Question:" in decoded:
        return decoded.split("Question:")[-1].split("\n", 1)[-1].strip()

    return decoded

# === Flask App Setup ===
app = Flask(__name__)
CORS(app)

@app.route("/rag", methods=["POST"])
def rag_endpoint():
    data = request.json
    query = data.get("query")
    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        # Classify query
        result = classifier(query, candidate_labels=labels)
        predicted_label = result["labels"][0]
        score = result["scores"][0]

        # Confidence threshold (tweakable)
        THRESHOLD = 0.75

        if predicted_label == "small talk" or (predicted_label == "domain question" and score < THRESHOLD):
            response = generate_small_talk_reply(query)
            return jsonify({
                "query": query,
                "context": None,
                "response": response
            })

        # RAG flow
        chunks = retrieve_chunks(query)
        context = "\n\n".join(chunks)
        response = generate_response(query, context)
        return jsonify({
            "query": query,
            "context": context,
            "response": response
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# === Run Server ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
