import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import get_hf_token

MODEL_LIST = [
    # "meta-llama/Llama-2-7b-hf",
    # "deepseek-ai/deepseek-llm-7b-chat",
    "Qwen/Qwen3-8B",
    # "gpt2"
]


def download_model_and_tokenizer(model_name: str, cache_dir: str, hf_token: str):
    """Download one model and its tokenizer."""
    print(f"\n{'='*60}")
    print(f"Starting download for: {model_name}")
    print(f"{'='*60}")
    print(f"Step 1/2: Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        token=hf_token,
        trust_remote_code=True
    )
    print("Tokenizer downloaded successfully.")

    print(f"Step 2/2: Downloading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        token=hf_token,
        trust_remote_code=True,
        torch_dtype="auto"
    )
    print("Model downloaded successfully.")

    return tokenizer, model


if __name__ == "__main__":
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Using cache directory: {cache_dir}")

    total_models = len(MODEL_LIST)
    print(f"\nFound {total_models} models to download. Starting batch process...")
    
    for idx, model_name in enumerate(MODEL_LIST, 1):
        try:
            download_model_and_tokenizer(
                model_name=model_name,
                cache_dir=cache_dir,
                hf_token=get_hf_token()
            )
            print(f"[{idx}/{total_models}] Download completed for: {model_name}\n")
        except Exception as e:
            print(f"[{idx}/{total_models}] Download failed for: {model_name}")
            print(f"Error details: {str(e)}\n")

    print(f"Batch download process finished. All models are saved to: {cache_dir}")
