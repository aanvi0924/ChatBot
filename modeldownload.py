import os

# Always get the directory where the current script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths relative to the script location
MODEL_DIR = os.path.join(SCRIPT_DIR, "models", "miniLM")
MISTRAL_DIR = os.path.join(SCRIPT_DIR, "finetuningllm", "mistral-7b-instruct")
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model.save(MODEL_DIR)
print(model.encode("This is a test sentence").shape)
from huggingface_hub import snapshot_download
os.makedirs(MISTRAL_DIR, exist_ok=True)

snapshot_download(
    repo_id="unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    local_dir=MISTRAL_DIR,
    local_dir_use_symlinks=False
)
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MISTRAL_DIR,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True
)

input_text = "what comes after alphabet a"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=20)

print("Model loaded successfully!")
print("Output:", tokenizer.decode(outputs[0]))
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MISTRAL_DIR,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True
)

input_text = "what comes after alphabet a"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=20)

print("Model loaded successfully!")
print("Output:", tokenizer.decode(outputs[0]))
