from unsloth import FastLanguageModel
import torch

max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/home/drut/chatbot/finetuningllm/mistral-7b-instruct",  #local model folder
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)
import json

# === Step 1: Load original JSON file (list of prompt/completion dicts) ===
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# === Step 2: Convert to ShareGPT-style format with "messages" key ===
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

# === Step 3: Save to .jsonl (one chat per line) ===
def save_as_jsonl(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# === File paths ===
input_path = "/home/drut/chatbot/finetuningllm/data/dsp_orchestration_full_prompts.json"
output_path = "/home/drut/chatbot/finetuningllm/data/dsp_orchestration_sharegpt_chat.jsonl"

# === Run everything ===
data = load_json(input_path)
converted = convert_to_sharegpt_chat_format(data)
save_as_jsonl(converted, output_path)

print(f" Converted and saved to:\n{output_path}")
import json
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template

# === Step 1: Load the ShareGPT-format dataset ===
dataset = load_dataset(
    "json",
    data_files="/home/drut/chatbot/finetuningllm/data/dsp_orchestration_sharegpt_chat.jsonl",
    split="train"
)

# === Step 2: Apply Unsloth's chat template to tokenizer ===
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

# === Step 3: Format the messages into training-ready text ===
def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [
        tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        for convo in convos
    ]
    return {"text": texts}

# === Step 4: Apply formatting to dataset ===
dataset = dataset.map(formatting_prompts_func, batched=True)

# === Step 5: Save the formatted dataset to .jsonl ===
output_path = "/home/drut/chatbot/finetuningllm/data/dsp_orchestration_sharegpt_formatted.jsonl"

with open(output_path, "w", encoding="utf-8") as f:
    for example in dataset:
        f.write(json.dumps(example, ensure_ascii=False) + "\n")

print(f" Formatted dataset saved to:\n{output_path}")
dataset[5]["messages"]
print(dataset[5]["text"])
unsloth_template = \
    "{{ bos_token }}"\
    "{{ 'You are a helpful assistant to the user\n' }}"\
    "{% for message in messages %}"\
        "{% if message['role'] == 'user' %}"\
            "{{ '>>> User: ' + message['content'] + '\n' }}"\
        "{% elif message['role'] == 'assistant' %}"\
            "{{ '>>> Assistant: ' + message['content'] + eos_token + '\n' }}"\
        "{% endif %}"\
    "{% endfor %}"\
    "{% if add_generation_prompt %}"\
        "{{ '>>> Assistant: ' }}"\
    "{% endif %}"
unsloth_eos_token = "eos_token"

if False:
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = (unsloth_template, unsloth_eos_token,), # You must provide a template and EOS token
        mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
        map_eos_token = True, # Maps <|im_end|> to </s> instead
    )
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)
trainer_stats = trainer.train()

import torch
from transformers import BitsAndBytesConfig
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

# === Fix: Enable CPU offloading with BitsAndBytesConfig ===
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_enable_fp32_cpu_offload=True,  # Allows fallback to CPU
)

# === Load fine-tuned model (checkpoint-60) ===
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/home/drut/outputs/checkpoint-60",
    max_seq_length=2048,
    dtype=torch.float16,
    quantization_config=bnb_config,      # <== Add this
    device_map="auto",                   # <== Allow automatic placement
)

FastLanguageModel.for_inference(model)

# === Apply ChatML template for ShareGPT-style prompts ===
tokenizer = get_chat_template(
    tokenizer,
    chat_template="chatml",
    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
    map_eos_token=True,
)

# === Run interactive inference loop ===
print("Model loaded. Type your question (or 'exit' to quit):")

while True:
    user_input = input("\n Your question: ").strip()
    if user_input.lower() in {"exit", "quit"}:
        print("Exiting inference.")
        break

    messages = [{"from": "human", "value": user_input}]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=100,
            do_sample=False,
            use_cache=True,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = decoded.split(user_input)[-1].strip()
    print(f"\n Model Response:\n{response}")



