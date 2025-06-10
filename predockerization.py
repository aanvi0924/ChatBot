import json
import os

def update_model_path():
    # Relative path to the JSON file from this script
    relative_json_path = "outputs/checkpoint-60/adapter_config.json"
    json_path = os.path.abspath(relative_json_path)

    old_path = "/home/drut/chat-bot/finetuningllm/mistral-7b-instruct"
    new_path = "/app/finetuningllm/mistral-7b-instruct"

    if not os.path.isfile(json_path):
        print(f" File not found: {json_path}")
        return

    # Load the JSON config
    with open(json_path, "r") as f:
        config = json.load(f)

    # Check and update the path
    if config.get("base_model_name_or_path") == old_path:
        config["base_model_name_or_path"] = new_path
        with open(json_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Updated path in: {relative_json_path}")
    else:
        print(" Path not updated. Either already correct or not matching the expected old path.")

if __name__ == "__main__":
    update_model_path()
