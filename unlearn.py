import json
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from peft import PeftModel, LoraConfig, get_peft_model
import os
from accelerate import Accelerator
import logging
import argparse

from prepdata import data_preprocess
from argsetting import parser_unlearn
from UnlearnTrainer import UnlearningTrainer, customize_collate_fn, CustomTripleDataset

class UnlearnQA(data_preprocess):
    def __init__(self, train_args, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.Load_RetainSet = False if (train_args.unlearn_method == "grad_ascent") else True
        self.Load_IdkSet = True if train_args.unlearn_method in ["dpo", "po"] else False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load dataset and create tokenized dataset
        self.forget_set = self.load_dataset(dataDIR = train_args.forgetSetDir)
        self.forget_set = self.tokenize_datasetQA(qa_data = self.forget_set)

        self.retain_set = None
        if self.Load_RetainSet:
            self.retain_set = self.load_dataset(dataDIR = train_args.retainSetDir)        
            self.retain_set = self.tokenize_datasetQA(qa_data = self.retain_set)    

        self.idk_set = None
        if self.Load_IdkSet:
            self.idk_set = self.load_idk_dataset(dataDIR = train_args.forgetSetDir, idkDIR = train_args.idkSetDir)        
            self.idk_set = self.tokenize_datasetQA(qa_data = self.idk_set)

        self.dataset = CustomTripleDataset(self.forget_set, self.retain_set, self.idk_set)
        print(f'[checkpoint]Load Dataset:{self.dataset}')
        
        # ----------------------------
        # Load source model
        # source_model_type:
        #   - learned: base model + finetuned LoRA adapter -> merge -> attach new unlearn LoRA
        #   - pt     : full pretrained checkpoint          -> full-parameter unlearning (NO LoRA)
        # ----------------------------
        self.source_model_type = train_args.source_model_type
        self.use_lora_unlearn = (self.source_model_type == "learned")
        
        if train_args.source_model_type == "learned":
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16
            )
            source_model = PeftModel.from_pretrained(
                base_model,
                train_args.finetune_model_DIR,
                local_files_only=True
            )
            source_model = source_model.merge_and_unload()
            print(f"[checkpoint] Load learned source model from {train_args.finetune_model_DIR}")

            # UserWarning: Already found a `peft_config` attribute in the model. 
            # This will lead to having multiple adapters in the model. 
            # Make sure to know what you are doing!
            lora_config = LoraConfig(
                r=train_args.LoRA_rank,
                lora_alpha=2 * train_args.LoRA_rank,
                lora_dropout=train_args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(source_model, lora_config)
            print("[checkpoint] Unlearning mode: LoRA adapter training on learned source model")
            
        elif train_args.source_model_type == "pt":
            self.model = AutoModelForCausalLM.from_pretrained(
                train_args.finetune_model_DIR,
                torch_dtype=torch.bfloat16,
                local_files_only=True
            )
            print(f"[checkpoint] Load pretrained source model from {train_args.finetune_model_DIR}")
            print("[checkpoint] Unlearning mode: full-parameter training on pt source model (NO LoRA)")

            for p in self.model.parameters():
                p.requires_grad = True

        else:
            raise ValueError(f"Unknown source_model_type: {train_args.source_model_type}")
        
        

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
            remove_unused_columns = False #if self.Load_RetainSet else True, # Set this to False when handling the customized dataset, otherwise dictionaries with customized keys could be deleted.
        )
        
        
        trainer = UnlearningTrainer(
            unlearn_method=train_args.unlearn_method,
            Load_RetainSet=self.Load_RetainSet,
            Load_IdkSet=self.Load_IdkSet,
            reg_weights=train_args.reg_weights,
            beta=train_args.beta,
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            eval_dataset=self.dataset,  
            # tokenizer=self.tokenizer,            
            data_collator=customize_collate_fn
        )

        # Add callbacks
        trainer.add_callback(EpochCheckpointCallback(
            save_every=4,
            base_path=train_args.unlearn_model_DIR
        ))

        return trainer


class EpochCheckpointCallback(TrainerCallback):
    def __init__(self, save_every=2, base_path="checkpoints"):
        self.save_every = save_every
        self.base_path = base_path

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        current_epoch = int(state.epoch)
        if current_epoch % self.save_every == 0:
            save_path = os.path.join(self.base_path, f"epoch-{current_epoch}")
            os.makedirs(save_path, exist_ok=True)

            # learned 路线下，model 是 PeftModel，save_pretrained 保存 adapter
            # pt 路线下，model 是普通 CausalLM，save_pretrained 保存完整 checkpoint
            kwargs["model"].save_pretrained(save_path)
            print(f"✅ Saved model at {save_path}")


def main():
    parse = parser_unlearn()
    train_args = parse.parse_args()
    
    # Create folders for saving the logger file and the unlearned model
    savefolder = f"{train_args.unlearnSet}-lr{train_args.lr}_WD{train_args.weight_decay}_loraRank{train_args.LoRA_rank}_loraDrop{train_args.lora_dropout}_GradStep{train_args.grad_acc_steps}_reg{train_args.reg_weights}"
    if train_args.beta != 0.1:
        savefolder += f"_beta{train_args.beta}"
    savefolder += f"/{train_args.unlearn_method}"
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
    
    # ----------------------------
    # Folder for loading the source model:
    #   learned  -> fine-tuned LoRA adapter directory
    #   pt       -> pretrained full-checkpoint directory
    # ----------------------------
    savefolder_tmp = (
        f"lr{train_args.lr_ft}_WD{train_args.wd_ft}_"
        f"loraRank{train_args.LoRA_rank_ft}_loraDrop{train_args.lora_dropout_ft}_"
        f"GradStsp{train_args.grad_acc_steps_ft}/epoch-{train_args.eps_ft}"
    )
    train_args.finetune_model_DIR = os.path.join(train_args.finetune_model_DIR, savefolder_tmp)
    print(f"[checkpoint] source_model_type={train_args.source_model_type}")
    print(f"[checkpoint] source model dir: {train_args.finetune_model_DIR}")    
    #####################
    
    from saved_hf_key import HF_key  # Replace 'HF_key' by your own hugging face key.
    os.environ["HF_TOKEN"] = HF_key
    
    if train_args.datasetName == 'FPI':
        file_path = "./data_generator/data"
        set_path = train_args.unlearnSet
        train_args.forgetSetDir = os.path.join(file_path, set_path, train_args.forgetSetDir)
        train_args.retainSetDir = os.path.join(file_path, set_path, train_args.retainSetDir)
        train_args.idkSetDir = os.path.join(file_path, train_args.idkSetDir)


    unleaner = UnlearnQA( 
                            train_args = train_args,
                            model_name = train_args.model_name, 
                            auth_token = HF_key,
                            )

    trainer = unleaner.BuildTrainer(train_args)
    unleaner.model, trainer = accelerator.prepare(unleaner.model, trainer)

    trainer.train()
    # learned 路线：保存 LoRA adapter
    # pt 路线：保存完整模型
    trainer.save_model(train_args.unlearn_model_DIR)

    for obj in trainer.state.log_history:
        logger.info(str(obj))

if __name__ == "__main__":
    main()    