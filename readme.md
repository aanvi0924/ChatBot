# LLM Project Workflow
This project demonstrates a two-stage workflow for working with LLMs: embedding and finetuning, each in its own Python virtual environment.

## Workflow Overview

1. **Embedding Stage**
    - Create a virtual environment.
    - Install dependencies from `requirements.txt`.
    - Run `modeldownload.py` to download and save embedding and LLM models.
    - (Optionally) Run your embedding script.

2. **Finetuning Stage**
    - Create a new, separate virtual environment.
    - Install dependencies from `requirements_finetuning.txt`.
    - Run `data_finetuning.py` to prepare data.
    - Run `finetuningllm.py` to finetune the LLM.
    - Run `inference.py` to test the finetuned model.

## Step-by-Step Instructions

### 1. Embedding Environment
```sh
python3 -m venv embedding_env
source embedding_env/bin/activate
pip install -r requirements.txt
python modeldownload.py
# python embedding.py   # (if you have an embedding script)
deactivate
```

### 2. Finetuning Environment
```sh
python3 -m venv finetuning_env
source finetuning_env/bin/activate
pip install -r requirements_finetuning.txt
python data_finetuning.py
python finetuningllm.py
python inference.py
deactivate
```

### 3. RAG Environment
```sh
python3 -m venv rag_env
source rag_env/bin/activate
pip install -r requirement_rag.txt
python rag_server.py
python predockerization.py
deactivate
```

## File Descriptions
- `requirements.txt` — Dependencies for embedding/model download.
- `modeldownload.py` — Downloads and saves embedding and LLM models.
- `requirements_finetuning.txt` — Dependencies for finetuning.
- `data_finetuning.py` — Prepares data for finetuning.
- `finetuningllm.py` — Finetunes the LLM.
- `inference.py` — Runs inference with the finetuned model.

## Notes
- Always activate the correct virtual environment before running scripts.
- Adjust paths in scripts as needed for your directory structure.
- Ensure you have sufficient disk space and GPU resources for model downloads and finetuning.

## Dockerization
After preparing your environment and running the necessary scripts, you can containerize your RAG server using Docker.

### Build the Docker Image
```sh
docker build -t chat-bot .
```
### Run the Docker Container
```sh
sudo docker container run --name=bot -dit --ipc=host --restart=always --gpus all -p 5000:5000 <name of the  dockerimage>
```
This will start the Flask server inside the container, exposing it on port 5000.
### Example API Request
You can test the running server with:
```sh
curl -X POST http://localhost:5000/rag -H "Content-Type: application/json" -d '{"query": "Your question here"}'
```

## docker image Structure
- `rag_server.py` — Main Flask API server
- `requirement_rag.txt` — Python dependencies
- `data/` — Data files and vector database
- `models/` — Pretrained embedding and LLM models
- `outputs/` — Model checkpoints
