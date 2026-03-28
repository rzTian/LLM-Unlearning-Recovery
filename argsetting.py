import argparse


def parser_finetune():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetName', default='FPI', type=str)
    parser.add_argument('--dataDIR', default="training_dataset.json" , type=str)
    parser.add_argument('--logDIR', default="fine_tuned_deepseek_7b_log", type=str)
    parser.add_argument('--modelDIR', default="fine_tuned_deepseek_7b", type=str)
    parser.add_argument('--model_name', default="deepseek-ai/deepseek-llm-7b-chat", type=str)    
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--LoRA_rank', default=32, type=int)
    parser.add_argument('--lora_dropout', default=0.0, type=float)
    # Control effective batch size
    parser.add_argument('--bs_train', default=8, type=int) # per device
    parser.add_argument('--bs_eval', default=8, type=int) # per device
    parser.add_argument('--grad_acc_steps', default=40, type=int)
    
    return parser



def parser_eval():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetName', default='FPI', type=str)
    parser.add_argument('--datasetType', default="forget", type=str)
    parser.add_argument('--unlearnSet', default="", type=str)
    parser.add_argument('--modelType', default='unlearned', type=str, choices=['base', 'learned', 'unlearned'])
    parser.add_argument('--model_name', default="deepseek-ai/deepseek-llm-7b-chat", type=str)
    parser.add_argument('--logDIR', default="fine_tuned_deepseek_7b_log", type=str)
    parser.add_argument('--modelDIR', default="fine_tuned_deepseek_7b", type=str)
    parser.add_argument('--unlearn_method', default="grad_diff" , type=str, choices=["grad_ascent", "grad_diff", "KL", "po", "dpo", "npo"])
    parser.add_argument('--quant', type=str, default="none", choices=["none", "int8", "int4"], help="Quant: none (Default)、int8、int4")
    # Finetuned model configs
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--LoRA_rank', default=256, type=int)
    parser.add_argument('--lora_dropout', default=0.0, type=float)
    parser.add_argument('--grad_acc_steps', default=40, type=int)
    # Unlearned model configs
    parser.add_argument('--num_fgt', default=1, type=int)
    parser.add_argument('--lr_fgt', default=0.001, type=float)
    parser.add_argument('--eps_fgt', default=0, type=int)
    parser.add_argument('--reg_weights_fgt', default=1.0, type=float)
    parser.add_argument('--wd_fgt', default=0.0, type=float)
    parser.add_argument('--LoRA_rank_fgt', default=256, type=int)
    parser.add_argument('--lora_dropout_fgt', default=0.0, type=float)
    parser.add_argument('--grad_acc_steps_fgt', default=80, type=int)
    parser.add_argument('--beta_fgt', default=0.1, type=float)
    parser.add_argument('--logDIR_fgt', default="unlearn_deepseek_7b_log", type=str)
    parser.add_argument('--modelDIR_fgt', default="unlearn_deepseek_7b", type=str)
    # generation configuration
    parser.add_argument('--max_new_tokens', default=100, type=int)
    parser.add_argument('--temperature', default=0.3, type=float)
    parser.add_argument('--top_p', default=0.5, type=float)
    return parser


def parser_unlearn():
    parser = argparse.ArgumentParser()    
    # unlearning methods
    parser.add_argument('--unlearn_method', default="grad_diff" , type=str, choices=["grad_ascent", "grad_diff", "KL", "po", "dpo", "npo"])
    # Training hyper-parameters
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--epochs', default=15, type=int)  ## Note that the epochs is w.r.t the size of the retain set.
    parser.add_argument('--reg_weights', default=1.0, type=float)
    parser.add_argument('--beta', default=0.1, type=float)
    parser.add_argument('--LoRA_rank', default=256, type=int)
    parser.add_argument('--lora_dropout', default=0.0, type=float)
    # Control effective batch size
    parser.add_argument('--bs_train', default=1, type=int) # per device
    parser.add_argument('--bs_eval', default=1, type=int) # per device
    parser.add_argument('--grad_acc_steps', default=80, type=int)
    # File name of the dataset
    parser.add_argument('--num_fgt', default=2025, type=int)
    parser.add_argument("--unlearnSet", default="unlearn-N1", type=str)
    parser.add_argument('--forgetSetDir', default="forget.json" , type=str)
    parser.add_argument('--retainSetDir', default="retain.json" , type=str)
    parser.add_argument('--idkSetDir', default="idk.jsonl" , type=str)
    # directories for loading and saving the model
    parser.add_argument('--finetune_model_DIR', default="fine_tuned_deepseek_7b", type=str)
    parser.add_argument('--logDIR', default="unlearn_deepseek_7b_log", type=str)
    parser.add_argument('--unlearn_model_DIR', default="unlearn_deepseek_7b", type=str)
    parser.add_argument('--model_name', default="deepseek-ai/deepseek-llm-7b-chat", type=str)
    parser.add_argument('--datasetName', default='FPI', type=str)
    # configs related to the loaded finetuned model
    parser.add_argument('--lr_ft', default=0.0005, type=float)
    parser.add_argument('--eps_ft', default=30, type=int)
    parser.add_argument('--wd_ft', default=0.01, type=float)
    parser.add_argument('--LoRA_rank_ft', default=256, type=int)
    parser.add_argument('--lora_dropout_ft', default=0.0, type=float)
    parser.add_argument('--grad_acc_steps_ft', default=40, type=int)
    return parser

