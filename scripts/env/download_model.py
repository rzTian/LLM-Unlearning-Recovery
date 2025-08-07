import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from saved_hf_key import HF_key
from argsetting import parser_finetune

# Parse command line arguments
parse = parser_finetune()
args = parse.parse_args()
model_name = args.model_name

# Set cache directory
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
os.makedirs(cache_dir, exist_ok=True)

# Download model and tokenizer
print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    token=HF_key
)

print("Downloading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    token=HF_key
)

print(f"Model downloaded to: {cache_dir}")