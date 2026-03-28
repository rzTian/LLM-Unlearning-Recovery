import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoTokenizer, AutoModelForCausalLM
from saved_hf_key import HF_key

MODEL_LIST = [
    # "meta-llama/Llama-2-7b-hf",
    # "deepseek-ai/deepseek-llm-7b-chat",
    # "openai/gpt-oss-20b",
    # "Qwen/Qwen3-8B",
    "gpt2"
]


def download_model_and_tokenizer(model_name: str, cache_dir: str, hf_token: str):
    """单模型下载函数（复用原有逻辑，便于批量调用）"""
    # 下载Tokenizer
    print(f"\n{'='*60}")
    print(f"Starting download for: {model_name}")
    print(f"{'='*60}")
    print(f"Step 1/2: Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        token=hf_token,
        trust_remote_code=True  # 适配DeepSeek/Mistral等模型的专属Tokenizer
    )
    print(f"Tokenizer downloaded successfully.")

    # 下载Model
    print(f"Step 2/2: Downloading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        token=hf_token,
        trust_remote_code=True,  # 适配非标准架构模型
        torch_dtype="auto"       # 自动匹配模型推荐数据类型（减少显存占用）
    )
    print(f"Model downloaded successfully.")

    return tokenizer, model


if __name__ == "__main__":
    # 缓存目录配置（保持原有逻辑）
    # 默认~/.cache/huggingface/hub ~/projects/def-yymao/hsc/LLM-Unlearning-Recovery/llm_models
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Using cache directory: {cache_dir}")

    # 批量下载模型列表中的所有模型
    total_models = len(MODEL_LIST)
    print(f"\nFound {total_models} models to download. Starting batch process...")
    
    for idx, model_name in enumerate(MODEL_LIST, 1):
        try:
            download_model_and_tokenizer(
                model_name=model_name,
                cache_dir=cache_dir,
                hf_token=HF_key
            )
            print(f"✅ [{idx}/{total_models}] Download completed for: {model_name}\n")
        except Exception as e:
            print(f"❌ [{idx}/{total_models}] Download failed for: {model_name}")
            print(f"Error details: {str(e)}\n")

    print(f"Batch download process finished. All models are saved to: {cache_dir}")