import json
import random
from pathlib import Path

# === STEP 1: Set File Paths ===
# 🛠️ Replace these with the actual paths on your machine:
input_path = Path("/home/drut/chatbot/finetuningllm/data/dsp_orchestration_formatted.jsonl")
train_path = Path("/home/drut/chatbot/finetuningllm/data/train_manual.jsonl")
val_path = Path("/home/drut/chatbot/finetuningllm/data/val_manual.jsonl")

# === STEP 2: Load and Parse the JSONL File ===
with input_path.open("r") as f:
    lines = f.readlines()

# Parse each line into a JSON object
all_data = [json.loads(line) for line in lines]

# === STEP 3: Extract Prompts and Attach Index ===
categorized_data = [
    {"index": idx, "prompt": item["messages"][0]["content"], "data": item}
    for idx, item in enumerate(all_data)
]

# === STEP 4: Manually Select Prompts for Validation Set ===
# These were selected for topical diversity:
val_prompts = {
    "How should I configure my network VLANs and assign IP addresses?",
    "How do I access the DSP GUI and what are the default login credentials?"
}

# Separate into validation and training based on selected prompts
val_data = [item["data"] for item in categorized_data if item["prompt"] in val_prompts]
train_data = [item["data"] for item in categorized_data if item["prompt"] not in val_prompts]

# === STEP 5: Write to Output Files ===
with train_path.open("w") as f:
    for item in train_data:
        f.write(json.dumps(item) + "\n")

with val_path.open("w") as f:
    for item in val_data:
        f.write(json.dumps(item) + "\n")

print(f" Split complete: {len(train_data)} training and {len(val_data)} validation examples.")
print(f"Train file saved to: {train_path}")
print(f"Validation file saved to: {val_path}")
