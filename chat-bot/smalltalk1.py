import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"  # protobuf fix

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
CLASSIFIER_MODEL_DIR = os.path.join(BASE_DIR, "models", "bart-large-mnli")  # Downloaded manually

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
    return decoded.replace(prompt, "").strip()

# === RAG Response Generator ===
def generate_response(query, context, max_tokens=300):
    messages = [{"from": "human", "value": f"Context: {context}\n\nQuestion: {query}"}]
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
    return decoded.split(query)[-1].strip()

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

        if predicted_label == "small talk":
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
