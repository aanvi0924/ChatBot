import os

# Define the base path relative to the script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
MODEL_PATH = os.path.join(SCRIPT_DIR, "finetuningllm", "mistral-7b-instruct")
DATA_PATH = os.path.join(SCRIPT_DIR, "finetuningllm", "data")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "outputs")

# Files
PROMPT_JSON = os.path.join(DATA_PATH, "dsp_orchestration_full_prompts.json")
CHAT_JSONL = os.path.join(DATA_PATH, "dsp_orchestration_sharegpt_chat.jsonl")
FORMATTED_JSONL = os.path.join(DATA_PATH, "dsp_orchestration_sharegpt_formatted.jsonl")
from unsloth import FastLanguageModel
import torch

max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)
import json

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def convert_to_sharegpt_chat_format(data):
    return [
        {
            "messages": [
                {"from": "human", "value": example["prompt"].strip()},
                {"from": "gpt", "value": example["completion"].strip()}
            ]
        }
        for example in data
    ]

def save_as_jsonl(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

data = load_json(PROMPT_JSON)
converted = convert_to_sharegpt_chat_format(data)
save_as_jsonl(converted, CHAT_JSONL)

print(f"Converted and saved to:\n{CHAT_JSONL}")
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template

dataset = load_dataset(
    "json",
    data_files=CHAT_JSONL,
    split="train"
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="chatml",
    mapping={
        "role": "from",
        "content": "value",
        "user": "human",
        "assistant": "gpt"
    },
    map_eos_token=True
)

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [
        tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        for convo in convos
    ]
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

with open(FORMATTED_JSONL, "w", encoding="utf-8") as f:
    for example in dataset:
        f.write(json.dumps(example, ensure_ascii=False) + "\n")

print(f"Formatted dataset saved to:\n{FORMATTED_JSONL}")
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=OUTPUT_PATH,
        report_to="none",
    ),
)

trainer_stats = trainer.train()
