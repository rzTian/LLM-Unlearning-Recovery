import json
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import PeftModel, LoraConfig, get_peft_model
import os
from accelerate import Accelerator
import logging
import argparse

from prepdata import data_preprocess
from argsetting import parser_unlearn
from UnlearnTrainer import UnlearningTrainer, customize_collate_fn, CustomDualDataset

class UnlearnQA(data_preprocess):
    def __init__(self, train_args, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.Load_RetainSet = False if (train_args.unlearn_method == "grad_ascent") else True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load dataset and create tokenized dataset
        self.forget_set = self.load_dataset(dataDIR = train_args.forgetSetDir)
        self.forget_set = self.tokenize_datasetQA(qa_data = self.forget_set)

        if self.Load_RetainSet:
            self.retain_set = self.load_dataset(dataDIR = train_args.retainSetDir)        
            self.retain_set = self.tokenize_datasetQA(qa_data = self.retain_set)        
            self.dual_dataset = CustomDualDataset(tokenized_ForgetSet = self.forget_set, tokenized_RetainSet = self.retain_set)
        
        # Load model
        base_model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.bfloat16)
        self.model = PeftModel.from_pretrained(base_model, train_args.finetune_model_DIR)
        self.model.merge_and_unload()

        lora_config = LoraConfig(r=train_args.LoRA_rank, 
                                    lora_alpha=2*train_args.LoRA_rank, 
                                    lora_dropout=train_args.lora_dropout, 
                                    bias="none", 
                                    task_type="CAUSAL_LM")

        self.model = get_peft_model(self.model, lora_config) 
        
        

    def BuildTrainer(self, train_args):

        training_args = TrainingArguments( 
            optim="adamw_torch",          
            output_dir=train_args.unlearn_model_DIR,
            eval_strategy="epoch",
            save_strategy="best",  
            save_total_limit=1,
            load_best_model_at_end=True,
            learning_rate=train_args.lr,
            lr_scheduler_type="reduce_lr_on_plateau",
            per_device_train_batch_size=train_args.bs_train,  
            per_device_eval_batch_size=train_args.bs_eval,
            gradient_accumulation_steps=train_args.grad_acc_steps,
            num_train_epochs=train_args.epochs,
            weight_decay=train_args.weight_decay,                       
            # fp16=True,
            bf16=True,
            push_to_hub=False,
            report_to="none",  
            logging_dir=train_args.unlearn_model_DIR,
            logging_first_step = True,
            logging_steps=1, 
            prediction_loss_only=True,  # Prevents logits from being stored, save memory. Otherwise may cause OOM.
            label_names=["labels"],
            remove_unused_columns = False if self.Load_RetainSet else True, # Set this to False when handling the customized dataset, otherwise dictionaries with customized keys could be deleted.
        )
        
        
        trainer = UnlearningTrainer(
            unlearn_method=train_args.unlearn_method,
            Load_RetainSet=self.Load_RetainSet,
            reg_weights=train_args.reg_weights,
            model=self.model,
            args=training_args,
            train_dataset=self.dual_dataset if self.Load_RetainSet else self.forget_set,
            eval_dataset=self.dual_dataset if self.Load_RetainSet else self.forget_set,  
            # tokenizer=self.tokenizer,            
            data_collator=customize_collate_fn if self.Load_RetainSet else None,
        )

        return trainer




def main():
    parse = parser_unlearn()
    train_args = parse.parse_args()
    
    # Create folders for saving the logger file and the unlearned model
    savefolder = f"fgt_profile-{train_args.num_fgt_prof}_attr-{train_args.num_fgt_attr}-lr{train_args.lr}_WD{train_args.weight_decay}_loraRank{train_args.LoRA_rank}_loraDrop{train_args.lora_dropout}_eps{train_args.epochs}_reg{train_args.reg_weights}_{train_args.unlearn_method}"
    train_args.logDIR = os.path.join(train_args.logDIR, savefolder)
    os.makedirs(train_args.logDIR, exist_ok=True)
    
    train_args.unlearn_model_DIR = os.path.join(train_args.unlearn_model_DIR, savefolder)
    os.makedirs(train_args.unlearn_model_DIR, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(train_args.logDIR, "result.log")),
            logging.StreamHandler()
        ])

    # Enable distributed training
    accelerator = Accelerator(mixed_precision="bf16")  #####
    logger.info(f"Using {accelerator.num_processes} GPUs") 
    
    # Folder for loading the fine-tuned model
    savefolder_tmp = f"lr{train_args.lr_ft}_WD{train_args.wd_ft}_loraRank{train_args.LoRA_rank_ft}_loraDrop{train_args.lora_dropout_ft}"
    train_args.finetune_model_DIR = os.path.join(train_args.finetune_model_DIR, savefolder_tmp)
    
    #####################
    
    from saved_hf_key import HF_key  # Replace 'HF_key' by your own hugging face key.
    os.environ["HF_TOKEN"] = HF_key
    
    if train_args.datasetName == 'FPI':
        train_args.forgetSetDir = f"forget-N_{train_args.num_fgt_prof}-attr-{train_args.num_fgt_attr}.json"
        train_args.retainSetDir = f"retain-N_{train_args.num_fgt_prof}-attr-{train_args.num_fgt_attr}.json"
        file_path = "./data_generator/data"
        train_args.forgetSetDir = os.path.join(file_path, train_args.forgetSetDir)
        train_args.retainSetDir = os.path.join(file_path, train_args.retainSetDir)


    unleaner = UnlearnQA( 
                            train_args = train_args,
                            model_name = train_args.model_name, 
                            auth_token = HF_key,
                            )

    trainer = unleaner.BuildTrainer(train_args)
    unleaner.model, trainer = accelerator.prepare(unleaner.model, trainer)

    trainer.train()
    trainer.save_model(train_args.unlearn_model_DIR)

    for obj in trainer.state.log_history:
        logger.info(str(obj))

if __name__ == "__main__":
    main()    