import os
import json
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb
import torch
from transformers import BitsAndBytesConfig
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

# === Setup Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_DIR = os.path.join(BASE_DIR, "data", "chromadb")
EMBED_MODEL_DIR = os.path.join(BASE_DIR, "models", "miniLM")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "outputs", "checkpoint-60")

# === Load Embedder and DB ===
embedder = SentenceTransformer(EMBED_MODEL_DIR)
client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
collection = client.get_or_create_collection(name="dsp_docs")

def retrieve_chunks(query, top_k=3):
    embedding = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=embedding, n_results=top_k)
    return results["documents"][0]

# === Load Mistral Model ===
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
    decoded = re.sub(r"<\|.*?\|>", "", decoded)
    return decoded.strip()

# === Test Questions ===
test_questions = [
    "What is the recommended operating system for installing DSP Orchestration",
    "What hardware resources are recommended for running DSP Orchestration and its components?",
    "What Python packages are needed to set up the DSP environment?",
    "What is the purpose of the dsp_orc_snap_path variable in dsp-orc.yaml?",
    "What does the skip_allocation flag control in DSP configuration"
]

ground_truths = {
    "What is the recommended operating system for installing DSP Orchestration": "Ubuntu 20.04 is the recommended operating system for DSP Orchestration",
    "What hardware resources are recommended for running DSP Orchestration and its components?": "A bare metal server capable of running DSP Orchestration, DFM, Prometheus, Postgres, and Grafana is recommended.",
    "What Python packages are needed to set up the DSP environment?": "You need to install python3, python3-venv, and python3-pip.",
    "What is the purpose of the dsp_orc_snap_path variable in dsp-orc.yaml?": "It sets the path to the DSP Orchestration snapshot tarball.",
    "What does the skip_allocation flag control in DSP configuration": "It determines whether to use existing machines (true) or to create new VMs (false)"
}

# === Generate Test Data ===
ragas_test_data = []

for question in tqdm(test_questions):
    try:
        chunks = retrieve_chunks(question)
        context = "\n\n".join(chunks)
        answer = generate_response(question, context)

        entry = {
            "question": question,
            "contexts": chunks,
            "answer": answer.strip()
        }

        if question in ground_truths:
            entry["ground_truth"] = ground_truths[question]

        ragas_test_data.append(entry)

    except Exception as e:
        print(f"Failed for question: {question} — {e}")

# === Save to JSON ===
output_path = os.path.join(BASE_DIR, "ragas_test_data.json")
with open(output_path, "w") as f:
    json.dump(ragas_test_data, f, indent=2)

print(f"\nRagas test data saved to: {output_path}")
