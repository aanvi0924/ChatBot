import torch
from transformers import BitsAndBytesConfig
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_enable_fp32_cpu_offload=True,
)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/home/drut/outputs/checkpoint-60",
    max_seq_length=2048,
    dtype=torch.float16,
    quantization_config=bnb_config,
    device_map=0,
)

FastLanguageModel.for_inference(model)

# 🧠 Get actual device Unsloth loaded the model on
model_device = next(model.parameters()).device

tokenizer = get_chat_template(
    tokenizer,
    chat_template="chatml",
    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
    map_eos_token=True,
)

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
    ).to(model_device)  # ✅ ensure correct GPU

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
