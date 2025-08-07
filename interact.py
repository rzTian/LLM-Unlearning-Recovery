import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel
from argsetting import parser_eval
import os


def load_model_and_tokenizer(eval_args, modelDIR):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(eval_args.model_name)

    if eval_args.modelType == 'base':
        model = AutoModelForCausalLM.from_pretrained(eval_args.model_name, torch_dtype=torch.bfloat16, device_map="auto")
    elif eval_args.modelType == 'learned':
        base_model = AutoModelForCausalLM.from_pretrained(eval_args.model_name, torch_dtype=torch.bfloat16, device_map="auto")
        model = PeftModel.from_pretrained(base_model, modelDIR["learned"])
        print(f"[checkpoint] Load learned model from {modelDIR['learned']}")
    elif eval_args.modelType == 'unlearned':
        base_model = AutoModelForCausalLM.from_pretrained(eval_args.model_name, torch_dtype=torch.bfloat16, device_map="auto")
        model = PeftModel.from_pretrained(base_model, modelDIR["learned"])
        model.merge_and_unload()
        model = PeftModel.from_pretrained(model, modelDIR["unlearned"])
        print(f"[checkpoint] Load unlearned model from {modelDIR['unlearned']}")
    else:
        raise ValueError("Invalid modelType")

    model.to(device)
    model.eval()
    return model, tokenizer, device


def interactive_loop(model, tokenizer, device, gen_cfg):
    print(">>> 输入问题进行交互，输入 'exit' 退出：")

    # 设置 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        # print("[info] tokenizer.pad_token 未设置，自动使用 eos_token 作为 pad_token")

    while True:
        user_input = input("输入：")
        if user_input.lower().strip() == "exit":
            break
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model.generate(**inputs, generation_config=gen_cfg)
        answer = tokenizer.decode(output[0], skip_special_tokens=True)
        print("回答：", answer, "\n")


def main():
    parser = parser_eval()
    eval_args = parser.parse_args()

    from evaluate import extract_dir
    modelDIR, _ = extract_dir(eval_args)

    model, tokenizer, device = load_model_and_tokenizer(eval_args, modelDIR)
    
    gen_cfg = GenerationConfig(
        max_new_tokens=64,
        # temperature=eval_args.temperature,
        # top_p=eval_args.top_p,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
    )

    interactive_loop(model, tokenizer, device, gen_cfg)

if __name__ == "__main__":
    main()
