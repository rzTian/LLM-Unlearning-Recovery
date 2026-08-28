import torch
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, TrainerCallback
from peft import LoraConfig, get_peft_model
import os
from accelerate import Accelerator
import logging

from prepdata import data_preprocess
from argsetting import parser_finetune


class TrainerQA(data_preprocess):
    def __init__(self, train_args, num_processes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_processes = num_processes
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
        # Use bfloat16 to reduce NaN risk for Llama-style models.
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            token=self.auth_token,
            torch_dtype=torch.bfloat16,
            local_files_only=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        if train_args.without_lora:
            print("[checkpoint] Finetune mode: full-parameter training (NO LoRA)")    
        else:
            lora_config = LoraConfig(
                r=train_args.LoRA_rank,
                lora_alpha=2 * train_args.LoRA_rank,
                lora_dropout=train_args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(self.model, lora_config)
            print("[checkpoint] Finetune mode: LoRA")
             
        self.entire_data = self.load_dataset(dataDIR = train_args.dataDIR)
        # Tokenize once before training.
        self.entire_data = self.tokenize_datasetQA(qa_data = self.entire_data)
        

    def BuildTrainer(self, train_args): 
        training_args = TrainingArguments( 
            optim="adamw_torch", 
            output_dir=train_args.modelDIR, 
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
            bf16=True, 
            push_to_hub=False, 
            report_to="none", 
            logging_dir=train_args.modelDIR, 
            logging_first_step = True, 
            logging_steps=1, 
            prediction_loss_only=True, # Prevents logits from being stored, save memory. Otherwise may cause OOM. 
            label_names=["labels"], 
        ) 
        
        trainer = Trainer( 
            model=self.model, 
            args=training_args, 
            train_dataset=self.entire_data, 
            eval_dataset=self.entire_data, 
            tokenizer=self.tokenizer, 
            callbacks=[
                SaveEveryNEpochsCallback(save_every=5, output_dir=train_args.modelDIR)
                ], 
            ) 
        
        return trainer


class SaveEveryNEpochsCallback(TrainerCallback):
    def __init__(self, save_every=5, output_dir=None):
        self.save_every = save_every
        self.output_dir = output_dir

    def on_epoch_end(self, args, state, control, **kwargs):
        current_epoch = int(state.epoch)
        if current_epoch % self.save_every == 0:
            save_path = os.path.join(self.output_dir, f"epoch-{int(state.epoch)}")
            kwargs["model"].save_pretrained(save_path)
            if args.process_index == 0:
                print(f"Saved model at {save_path}")
        return control    


def main():
    parse = parser_finetune()
    train_args = parse.parse_args()

    savefolder = f"lr{train_args.lr}_WD{train_args.weight_decay}_loraRank{train_args.LoRA_rank}_loraDrop{train_args.lora_dropout}_GradStsp{train_args.grad_acc_steps}"
    logDIR = os.path.join(train_args.logDIR, savefolder)
    os.makedirs(logDIR, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(logDIR, "result.log")),
            logging.StreamHandler()
        ])

    accelerator = Accelerator(mixed_precision="bf16")
    logger.info(f"Using {accelerator.num_processes} GPUs") 

    train_args.modelDIR = os.path.join(train_args.modelDIR, savefolder)
    os.makedirs(train_args.modelDIR, exist_ok=True)

    from saved_hf_key import HF_key
    os.environ["HF_TOKEN"] = HF_key

    if train_args.datasetName == 'FPI':
        file_path = "./data_generator/data"
        train_args.dataDIR = os.path.join(file_path, train_args.dataDIR)
        
    

    trainQA = TrainerQA(
                        train_args = train_args,
                        num_processes=accelerator.num_processes,
                        model_name = train_args.model_name, 
                        auth_token = HF_key
                        )

    trainer = trainQA.BuildTrainer(train_args)
    # `Trainer` already initializes and wraps the model for distributed training.
    # Preparing both the raw model and the Trainer again can lead to rank-wise
    # parameter mismatches when running under `accelerate launch`.

    trainer.train()
    trainer.save_model(train_args.modelDIR)

    for obj in trainer.state.log_history:
        logger.info(str(obj))
    


if __name__ == "__main__":
    main()
