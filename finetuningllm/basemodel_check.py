# 1. Download the model directly to a local folder
from huggingface_hub import snapshot_download
import os

# Define a local path (on the Colab VM or your local machine
local_model_path = "mistral-7b-instruct"

# Make sure the folder exists
os.makedirs(local_model_path, exist_ok=True)

# Download model into that folder
snapshot_download(
    repo_id = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    local_dir = local_model_path,
    local_dir_use_symlinks = False
)

# 2. Load the model from the local path using Unsloth
from unsloth import FastLanguageModel
import torch

max_seq_length = 2048  # RoPE scaling auto handled
dtype = None           # Auto-detected: Float16 or BFloat16
load_in_4bit = True    # 4bit quantization for memory savings

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = local_model_path,   # Load from local path
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
import os
print(os.listdir("/home/drut/chatbot/finetuningllm/mistral-7b-instruct"))

from unsloth import FastLanguageModel

model_path = "/home/drut/chatbot/finetuningllm/mistral-7b-instruct"

try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = 2048,
        dtype = None,  # Or "auto"
        load_in_4bit = True  # Set to False if you didn't quantize
    )

    # Try generating something
    input_text = "what comes after alphabet a"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=20)

    print(" Model loaded successfully!")
    print(" Output:", tokenizer.decode(outputs[0]))

except Exception as e:
    print(" Error loading model:", e)
