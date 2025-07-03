# LLM Project Workflow
This project demonstrates a two-stage workflow for working with LLMs: embedding and finetuning, each in its own Python virtual environment.

## Overview

This project implements a modular workflow for building, finetuning, and deploying a Retrieval-Augmented Generation (RAG) pipeline using Large Language Models (LLMs). It is designed for flexibility, allowing users to swap models and data sources by editing configuration paths in `.env` files.

---

## Tech Stack

- **Python 3.10+**
- **PyTorch** — Deep learning framework for model training and inference
- **Transformers** — Hugging Face library for LLMs and tokenizers
- **Sentence-Transformers** — For generating embeddings
- **Datasets** — For handling and processing datasets
- **Accelerate** — For efficient model training
- **PEFT** — Parameter-efficient finetuning
- **FastAPI / Flask** — For serving the RAG pipeline as an API
- **Docker** — For containerization and deployment

---
## Use Cases

- **Enterprise Q&A bots** — Ingest internal documents and answer employee queries.
- **Customer support** — Automate responses using your own knowledge base.
- **Research assistants** — Retrieve and summarize information from large corpora.
- **Custom LLM applications** — Easily swap models and data for new domains.

---

## Workflow Overview

1. **Embedding Stage**
    - Create a virtual environment.
    - Install dependencies from `requirements.txt`.
    - Run `modeldownload.py` to download and save LLM models(all-MiniLM-L6-v2)for embeddings, (mistral-7b-instruct-v0.3-bnb-4bit)for llm finetuning, for classifying the given content is small talk or not (bart-large-mnli) 
    - Set the required .path in the `embedding.env` file for your `embedding.py` script.
    - Run your `embedding.py` script.

2. **Finetuning Stage**
    - Create a new, separate virtual environment.
    - Install dependencies from `requirements_finetuning.txt`.
    - Set the required .path in the `data_finetuning.env` file for your `data_finetuning.py` script.
    - Run `data_finetuning.py` to prepare data.
    - Run `finetuningllm.py` to finetune the LLM.
    - Run `inference.py` to test the finetuned model.

3. **RAG Stage**

- Create a new virtual environment.
- Install dependencies from `requirement_rag.txt`.
- (optional) Run `rag_pipeline.py` to initialize or update the retrieval-augmented generation pipeline.
- Start the API server by running` rag_server.py` to serve the RAG pipeline for external queries.
- Run `predockerization.py` to prepare the application for Docker containerization.


## Step-by-Step Instructions

### 1. Embedding Environment
```sh
python3 -m venv embedding_env
source embedding_env/bin/activate
pip install -r requirement.txt
python3 modeldownload.py
python3 embedding.py   
deactivate
```

### 2. Finetuning Environment
```sh
python3 -m venv finetuning_env
source finetuning_env/bin/activate
pip install -r requirement_finetuning.txt
python3 data_finetuning.py
python3 finetuningllm.py
python3 inference.py
deactivate
```

### 3. RAG Environment
```sh
python3 -m venv rag_env
source rag_env/bin/activate
pip install -r requirement_rag.txt
python3 rag_server.py
python3 predockerization.py
deactivate
```

## File Descriptions
- `requirements.txt` — Lists all Python dependencies needed for downloading and running embedding and LLM models, such as `torch`, `transformers`, and `sentence-transformers`.
- `modeldownload.py` — Contains scripts to download pre-trained embedding models and large language models (LLMs) from sources like Hugging Face, and saves them locally for later use.
- `embeddings.py` — Handles the creation, storage, and retrieval of vector embeddings for documents or queries, supporting efficient similarity search and retrieval in the RAG pipeline.
- `requirements_finetuning.txt` — Specifies additional Python dependencies required specifically for the finetuning process, such as `datasets`, `accelerate`, and `peft`.
- `data_finetuning.py` — Prepares and processes raw data for finetuning, including cleaning, formatting, tokenization, and splitting into training and validation sets.
- `finetuningllm.py` — Implements the logic for finetuning the downloaded LLM on custom datasets, including the training loop, evaluation, and saving the finetuned model.
- `inference.py` — Provides scripts to load the finetuned model and run inference (generate predictions or responses) on
- `rag_pipeline.py` — Implements the Retrieval-Augmented Generation (RAG) pipeline, combining retrieval of relevant documents using embeddings with LLM-based answer generation.
- `rag_server.py` — Sets up an API server (e.g., using FastAPI or Flask) to serve the RAG pipeline, allowing external applications to send queries and receive generated responses.
- `predockerization.py` — Contains scripts or configuration steps to prepare the application for containerization, such as environment setup, dependency checks, and file organization.


## Notes
- Always activate the correct virtual environment before running scripts.
- Adjust paths in scripts as needed for your directory structure.
- Ensure you have sufficient disk space and GPU resources for model downloads and finetuning.

## JSON Message Formats

### Original Format

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What are the hardware and software prerequisites for installing DSP Orchestration?"
    },
    {
      "role": "assistant",
      "content": "DSP Orchestration requires a bare metal server with Ubuntu 20.04 installed. The recommended hardware should support running DSP orchestration, DFM, Prometheus, Postgres, and Grafana on the same machine. For software, required dependencies include Python 3 and relevant Python virtual environment packages (python3-venv, python3-pip). The environment setup requires installing these packages and setting up the Python virtual environment named dsp_env."
    }
  ]
}
```

### Final Format

```json
{
  "messages": [
    {
      "from": "human",
      "value": "What are the hardware and software prerequisites for installing DSP Orchestration?"
    },
    {
      "from": "gpt",
      "value": "DSP Orchestration requires a bare metal server with Ubuntu 20.04 installed. The recommended hardware should support running DSP orchestration, DFM, Prometheus, Postgres, and Grafana on the same machine. For software, required dependencies include Python 3 and relevant Python virtual environment packages (python3-venv, python3-pip). The environment setup requires installing these packages and setting up the Python virtual environment named dsp_env."
    }
  ],
  "text": "<|im_start|>user\nWhat are the hardware and software prerequisites for installing DSP Orchestration?<|im_end|>\n<|im_start|>assistant\nDSP Orchestration requires a bare metal server with Ubuntu 20.04 installed. The recommended hardware should support running DSP orchestration, DFM, Prometheus, Postgres, and Grafana on the same machine. For software, required dependencies include Python 3 and relevant Python virtual environment packages (python3-venv, python3-pip). The environment setup requires installing these packages and setting up the Python virtual environment named dsp_env.<|im_end|>\n"
}
```
## Making the Project Editable

All important file and directory paths are set in `.env` files (e.g., `embedding.env`, `data_finetuning.env`).  
**To customize the workflow:**
1. Open the relevant `.env` file for the stage you want to modify.
2. Change the paths for models, data, or output directories as needed.
3. Run the scripts as usual — they will pick up your new configuration.

---

## Dockerization
After preparing your environment and running the necessary scripts, you can containerize your RAG server using Docker.

### Build the Docker Image
```sh
sudo docker build -t chat-bot .
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
