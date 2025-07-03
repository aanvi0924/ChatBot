import os
import json
import requests
from dotenv import load_dotenv

# === Load environment variables ===
load_dotenv("data_finetuning.env")
JSON_URL = os.getenv("JSON_URL")
if not JSON_URL:
    raise ValueError("JSON_URL environment variable not set.")

# === Set up directories relative to script location ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "finetuningllm", "data")
os.makedirs(DATA_DIR, exist_ok=True)

# === File paths ===
JSON_PATH = os.path.join(DATA_DIR, "dsp_orchestration_full_prompts.json")
JSONL_PATH = os.path.join(DATA_DIR, "dsp_orchestration_formatted.jsonl")

# === Download the JSON file ===
print(f"Downloading dataset from {JSON_URL}...")
response = requests.get(JSON_URL)
response.raise_for_status()
with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(response.json(), f, ensure_ascii=False, indent=2)
print(f"Downloaded JSON saved to: {os.path.relpath(JSON_PATH)}")

# === Convert to JSONL format ===
print("Converting JSON to JSONL format...")
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(JSONL_PATH, "w", encoding="utf-8") as outfile:
    for pair in data:
        messages = [
            {"role": "user", "content": pair["prompt"]},
            {"role": "assistant", "content": pair["completion"]}
        ]
        json.dump({"messages": messages}, outfile, ensure_ascii=False)
        outfile.write("\n")
print(f"Formatted JSONL saved to: {os.path.relpath(JSONL_PATH)}")

# === Validate JSONL structure ===
print("Validating JSONL structure...")
with open(JSONL_PATH, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, start=1):
        try:
            item = json.loads(line)
            assert isinstance(item, dict), f"Line {i} is not a JSON object."
            assert "messages" in item, f"Line {i} missing 'messages' key."
            assert isinstance(item["messages"], list), f"Line {i} 'messages' is not a list."
            for msg in item["messages"]:
                assert isinstance(msg, dict), f"Line {i} contains a non-dict message."
                assert "role" in msg and "content" in msg, f"Line {i} message missing 'role' or 'content'."
                assert msg["role"] in ["user", "assistant"], f"Line {i} has invalid role: {msg['role']}"
        except Exception as e:
            print(f"Invalid JSONL on line {i}: {e}")
            break
    else:
        print("All lines are valid JSONL and match the expected structure.")
