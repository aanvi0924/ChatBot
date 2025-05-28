import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"  # <- Avoid protobuf crash

import torch
import chromadb
from transformers import BitsAndBytesConfig
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from sentence_transformers import SentenceTransformer

#Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_DIR = os.path.join(BASE_DIR, "data", "chromadb")
EMBED_MODEL_DIR = os.path.join(BASE_DIR, "models", "miniLM")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "outputs", "checkpoint-60")

#Vector Search Setup 
embedder = SentenceTransformer(EMBED_MODEL_DIR)
client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
collection = client.get_or_create_collection(name="dsp_docs")

def retrieve_chunks(query, top_k=3):
    embedding = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=embedding, n_results=top_k)
    return results["documents"][0]

#Load LLM 
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

#RAG Inference 
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

#rag
def main():
    print(" RAG Assistant is running. Ask a question (type 'exit' to quit):")
    while True:
        query = input("\nYour question: ").strip()
        if query.lower() in {"exit", "quit"}:
            print("Exiting.")
            break
        try:
            chunks = retrieve_chunks(query)
            context = "\n\n".join(chunks)
            response = generate_response(query, context)
            ##print(f"\nContext:\n{context}\n")
            print(f" Answer:\n{response}")
        except Exception as e:
            print(f" Error: {e}")

if __name__ == "__main__":
    main()
