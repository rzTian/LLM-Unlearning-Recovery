import argparse


def parser_finetune():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetName', default='FPI', type=str)
    parser.add_argument('--dataDIR', default="training_dataset.json" , type=str)
    parser.add_argument('--logDIR', default="fine_tuned_llama_7b_log", type=str)
    parser.add_argument('--modelDIR', default="fine_tuned_llama_7b", type=str)
    parser.add_argument('--model_name', default="meta-llama/Llama-2-7b-hf", type=str)    
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--LoRA_rank', default=32, type=int)
    parser.add_argument('--lora_dropout', default=0.0, type=float)
    # Control effective batch size
    parser.add_argument('--bs_train', default=8, type=int) # per device
    parser.add_argument('--bs_eval', default=8, type=int) # per device
    parser.add_argument('--grad_acc_steps', default=4, type=int)
    
    return parser



def parser_eval():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetName', default='FPI', type=str)
    parser.add_argument('--datasetType', default="forget", type=str)
    parser.add_argument('--modelType', default='unlearned', type=str, choices=['base', 'learned', 'unlearned'])
    parser.add_argument('--model_name', default="meta-llama/Llama-2-7b-hf", type=str)
    parser.add_argument('--logDIR', default="fine_tuned_llama_7b_log", type=str)
    parser.add_argument('--unlearn_method', default="grad_diff" , type=str, choices=["grad_ascent", "grad_diff", "KL"])
    # Finetuned model configs
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--LoRA_rank', default=32, type=int)
    parser.add_argument('--lora_dropout', default=0.0, type=float)
    # Unlearned model configs
    parser.add_argument('--num_fgt', default=1, type=int)
    parser.add_argument('--lr_fgt', default=0.001, type=float)
    parser.add_argument('--eps_fgt', default=1, type=int)
    parser.add_argument('--reg_weights_fgt', default=1.0, type=float)
    parser.add_argument('--wd_fgt', default=0.0, type=float)
    parser.add_argument('--LoRA_rank_fgt', default=32, type=int)
    parser.add_argument('--lora_dropout_fgt', default=0.0, type=float)
    # generation configuration
    parser.add_argument('--max_new_tokens', default=17, type=int)
    parser.add_argument('--temperature', default=0.3, type=float)
    parser.add_argument('--top_p', default=0.5, type=float)
    return parser


def parser_unlearn():
    parser = argparse.ArgumentParser()    
    # unlearning methods
    parser.add_argument('--unlearn_method', default="grad_diff" , type=str, choices=["grad_ascent", "grad_diff", "KL"])
    # Training hyper-parameters
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--epochs', default=6, type=int)  ## Note that the epochs is w.r.t the size of the retain set.
    parser.add_argument('--reg_weights', default=1.0, type=float)
    parser.add_argument('--LoRA_rank', default=32, type=int)
    parser.add_argument('--lora_dropout', default=0.0, type=float)
    # Control effective batch size
    parser.add_argument('--bs_train', default=5, type=int) # per device
    parser.add_argument('--bs_eval', default=5, type=int) # per device
    parser.add_argument('--grad_acc_steps', default=1, type=int)
    # File name of the dataset
    parser.add_argument('--num_fgt', default=2025, type=int)
    parser.add_argument('--forgetSetDir', default="forget-attribute.json" , type=str)
    parser.add_argument('--retainSetDir', default="retain-attribute.json" , type=str)
    # directories for loading and saving the model
    parser.add_argument('--finetune_model_DIR', default="fine_tuned_llama_7b", type=str)
    parser.add_argument('--logDIR', default="unlearn_llama_7b_log", type=str)
    parser.add_argument('--unlearn_model_DIR', default="unlearn_llama_7b", type=str)
    parser.add_argument('--model_name', default="meta-llama/Llama-2-7b-hf", type=str)
    parser.add_argument('--datasetName', default='FPI', type=str)
    # configs related to the loaded finetuned model
    parser.add_argument('--lr_ft', default=0.001, type=float)
    parser.add_argument('--wd_ft', default=0.0, type=float)
    parser.add_argument('--LoRA_rank_ft', default=32, type=int)
    parser.add_argument('--lora_dropout_ft', default=0.0, type=float)
    return parser

