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

from transformers import AutoTokenizer, AutoModelForSequenceClassification
model_id = "facebook/bart-large-mnli"
model = AutoModelForSequenceClassification.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Save to disk
model.save_pretrained("./models/bart-large-mnli")
tokenizer.save_pretrained("./models/bart-large-mnli")
print(f"Model and tokenizer saved to ./models/bart-large-mnli")

import spacy.cli
import shutil
import os

# === Target model name and destination ===
model_name = "en_core_web_sm"
target_dir = os.path.join("models", model_name)

# Download the model if not already present
if not os.path.exists(target_dir):
    print(f" Downloading and installing spaCy model: {model_name}")
    spacy.cli.download(model_name)

    # Get installed model path
    installed_path = spacy.util.get_package_path(model_name)

    # Copy it to ./models/en_core_web_sm
    shutil.copytree(installed_path, target_dir)
    print(f" Model copied to: {target_dir}")
else:
    print(f" Model already exists at: {target_dir}")
