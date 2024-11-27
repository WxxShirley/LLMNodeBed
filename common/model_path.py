common_path = "YOUR PATH OF MODELs"

MODEL_PATHs = {
    # LM
    "MiniLM": f"{common_path}/models--sentence-transformers--all-MiniLM-L6-v2",
    "SentenceBert": f"{common_path}/models--sentence-transformers--multi-qa-distilbert-cos-v1/snapshots/SentenceBert",
    "e5-large": f"{common_path}/models--intfloat--e5-large-v2/snapshots/e5-large",
    "roberta": f"{common_path}/models--sentence-transformers--all-roberta-large-v1/snapshots/roberta",

    # LLM
    "Qwen-3B": f"{common_path}/models--Qwen--Qwen2.5-3B-Instruct/snapshots/Qwen2.5-3B", 
    "Qwen-7B": f"{common_path}/models--Qwen--Qwen2.5-7B-Instruct/snapshots/Qwen2.5-7B",
    "Mistral-7B": f"{common_path}/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/Mistral-7B",
    "Llama-8B": f"{common_path}/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/Llama-8B"
}
