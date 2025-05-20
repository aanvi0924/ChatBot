import json

with open("/home/drut/chatbot/finetuningllm/data/dsp_orchestration_full_prompts.json", "r") as f:
    data = json.load(f)

with open("/home/drut/chatbot/finetuningllm/data/dsp_orchestration_formatted.jsonl", "w") as outfile:
    for pair in data:
        messages = [
            {"role": "user", "content": pair["prompt"]},
            {"role": "assistant", "content": pair["completion"]}
        ]
        json.dump({"messages": messages}, outfile)
        outfile.write("\n")
import json

file_path = "/home/drut/chatbot/finetuningllm/data/dsp_orchestration_formatted.jsonl"

with open(file_path, "r") as f:
    for i, line in enumerate(f, start=1):
        try:
            data = json.loads(line)
            assert isinstance(data, dict), f"Line {i} is not a JSON object."
            assert "messages" in data, f"Line {i} missing 'messages' key."
            assert isinstance(data["messages"], list), f"Line {i} 'messages' is not a list."
            for msg in data["messages"]:
                assert isinstance(msg, dict), f"Line {i} contains a non-dict message."
                assert "role" in msg and "content" in msg, f"Line {i} message missing 'role' or 'content'."
                assert msg["role"] in ["user", "assistant"], f"Line {i} has invalid role: {msg['role']}"
        except Exception as e:
            print(f" Invalid JSONL on line {i}: {e}")
            break
    else:
        print(" All lines are valid JSONL and match the expected structure.")
