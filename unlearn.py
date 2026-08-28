import logging
import os

import torch
from accelerate import Accelerator
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, TrainerCallback, TrainingArguments

from argsetting import parser_unlearn
from prepdata import data_preprocess
from UnlearnTrainer import UnlearningTrainer, customize_collate_fn, CustomTripleDataset


def get_unlearn_savefolder(train_args):
    savefolder = (
        f"{train_args.unlearnSet}-lr{train_args.lr}_WD{train_args.weight_decay}_"
        f"loraRank{train_args.LoRA_rank}_loraDrop{train_args.lora_dropout}_"
        f"GradStep{train_args.grad_acc_steps}_reg{train_args.reg_weights}"
    )
    if train_args.beta != 0.1:
        savefolder += f"_beta{train_args.beta}"

    if train_args.unlearn_method == "noisy_grad_diff":
        savefolder += f"_nstd{train_args.noisy_noise_std}_nclip{train_args.noisy_clip_norm}"

    savefolder += f"/{train_args.unlearn_method}"
    return savefolder


class UnlearnQA(data_preprocess):
    def __init__(self, train_args, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.Load_RetainSet = False if (train_args.unlearn_method == "grad_ascent") else True
        self.Load_IdkSet = True if train_args.unlearn_method in ["dpo", "po"] else False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.forget_raw = self.load_dataset(dataDIR=train_args.forgetSetDir)
        self.retain_raw = self.load_dataset(dataDIR=train_args.retainSetDir) if self.Load_RetainSet else None

        self.forget_set = self.tokenize_datasetQA(qa_data=self.forget_raw)
        self.retain_set = self.tokenize_datasetQA(qa_data=self.retain_raw) if self.retain_raw is not None else None

        self.idk_set = None
        if self.Load_IdkSet:
            self.idk_set = self.load_idk_dataset(dataDIR=train_args.forgetSetDir, idkDIR=train_args.idkSetDir)
            self.idk_set = self.tokenize_datasetQA(qa_data=self.idk_set)

        self.dataset = CustomTripleDataset(self.forget_set, self.retain_set, self.idk_set)
        print(f"[checkpoint]Load Dataset:{self.dataset}")

        self.source_model_type = train_args.source_model_type
        self.use_lora_unlearn = self.source_model_type == "learned"

        if train_args.source_model_type == "learned":
            base_model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.bfloat16)
            source_model = PeftModel.from_pretrained(base_model, train_args.finetune_model_DIR, local_files_only=True)
            source_model = source_model.merge_and_unload()
            print(f"[checkpoint] Load learned source model from {train_args.finetune_model_DIR}")

            lora_config = LoraConfig(
                r=train_args.LoRA_rank,
                lora_alpha=2 * train_args.LoRA_rank,
                lora_dropout=train_args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(source_model, lora_config)
            print("[checkpoint] Unlearning mode: LoRA adapter training on learned source model")

        elif train_args.source_model_type == "pt":
            self.model = AutoModelForCausalLM.from_pretrained(
                train_args.finetune_model_DIR,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )
            print(f"[checkpoint] Load pretrained source model from {train_args.finetune_model_DIR}")
            print("[checkpoint] Unlearning mode: full-parameter training on pt source model (NO LoRA)")
            for p in self.model.parameters():
                p.requires_grad = True
        else:
            raise ValueError(f"Unknown source_model_type: {train_args.source_model_type}")

    def build_standard_trainer(self, train_args):
        training_args = self._make_training_args(train_args)
        trainer = UnlearningTrainer(
            unlearn_method=train_args.unlearn_method,
            Load_RetainSet=self.Load_RetainSet,
            Load_IdkSet=self.Load_IdkSet,
            reg_weights=train_args.reg_weights,
            beta=train_args.beta,
            noisy_noise_std=train_args.noisy_noise_std,
            noisy_clip_norm=train_args.noisy_clip_norm,
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            eval_dataset=self.dataset,
            data_collator=customize_collate_fn,
        )
        trainer.add_callback(EpochCheckpointCallback(save_every=2, base_path=train_args.unlearn_model_DIR))
        return trainer

    def _make_training_args(self, train_args):
        return TrainingArguments(
            optim="adamw_torch",
            output_dir=train_args.unlearn_model_DIR,
            eval_strategy="epoch",
            save_strategy="epoch",
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
            logging_dir=train_args.unlearn_model_DIR,
            logging_first_step=True,
            logging_steps=1,
            prediction_loss_only=True,
            label_names=["labels"],
            remove_unused_columns=False,
        )


class EpochCheckpointCallback(TrainerCallback):
    def __init__(self, save_every=2, base_path="checkpoints"):
        self.save_every = save_every
        self.base_path = base_path

    def on_epoch_end(self, args, state, control, **kwargs):
        current_epoch = int(state.epoch)
        if current_epoch % self.save_every == 0:
            save_path = os.path.join(self.base_path, f"epoch-{current_epoch}")
            os.makedirs(save_path, exist_ok=True)
            kwargs["model"].save_pretrained(save_path)
            print(f"Saved model at {save_path}")


def main():
    parse = parser_unlearn()
    train_args = parse.parse_args()

    savefolder = get_unlearn_savefolder(train_args)
    train_args.logDIR = os.path.join(train_args.logDIR, savefolder)
    train_args.unlearn_model_DIR = os.path.join(train_args.unlearn_model_DIR, savefolder)
    os.makedirs(train_args.logDIR, exist_ok=True)
    os.makedirs(train_args.unlearn_model_DIR, exist_ok=True)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="[%(asctime)s] - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(train_args.logDIR, "result.log")),
            logging.StreamHandler(),
        ],
    )

    accelerator = Accelerator(mixed_precision="bf16")
    logger.info(f"Using {accelerator.num_processes} GPUs")

    savefolder_tmp = (
        f"lr{train_args.lr_ft}_WD{train_args.wd_ft}_"
        f"loraRank{train_args.LoRA_rank_ft}_loraDrop{train_args.lora_dropout_ft}_"
        f"GradStsp{train_args.grad_acc_steps_ft}/epoch-{train_args.eps_ft}"
    )
    train_args.finetune_model_DIR = os.path.join(train_args.finetune_model_DIR, savefolder_tmp)
    print(f"[checkpoint] source_model_type={train_args.source_model_type}")
    print(f"[checkpoint] source model dir: {train_args.finetune_model_DIR}")

    from saved_hf_key import HF_key

    os.environ["HF_TOKEN"] = HF_key

    if train_args.datasetName == "FPI":
        file_path = "./data_generator/data"
        set_path = train_args.unlearnSet
        train_args.forgetSetDir = os.path.join(file_path, set_path, train_args.forgetSetDir)
        train_args.retainSetDir = os.path.join(file_path, set_path, train_args.retainSetDir)
        train_args.idkSetDir = os.path.join(file_path, train_args.idkSetDir)

    unleaner = UnlearnQA(train_args=train_args, model_name=train_args.model_name, auth_token=HF_key)

    trainer = unleaner.build_standard_trainer(train_args)
    trainer.train()
    trainer.save_model(train_args.unlearn_model_DIR)
    for obj in trainer.state.log_history:
        logger.info(str(obj))


if __name__ == "__main__":
    main()
