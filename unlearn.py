import logging
import os

import torch
from accelerate import Accelerator
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, TrainerCallback, TrainingArguments

from argsetting import parser_unlearn
from prepdata import data_preprocess
from UnlearnTrainer import UnlearningTrainer, customize_collate_fn, CustomTripleDataset

try:
    import dp_transformers
    import dp_transformers.dp_utils
except Exception:
    dp_transformers = None


def get_unlearn_savefolder(train_args):
    savefolder = (
        f"{train_args.unlearnSet}-lr{train_args.lr}_WD{train_args.weight_decay}_"
        f"loraRank{train_args.LoRA_rank}_loraDrop{train_args.lora_dropout}_"
        f"GradStep{train_args.grad_acc_steps}_reg{train_args.reg_weights}"
    )
    if train_args.beta != 0.1:
        savefolder += f"_beta{train_args.beta}"

    if train_args.unlearn_method in ("langevin", "dp_random_label"):
        savefolder += f"_noise{train_args.noise_multiplier}_clip{train_args.max_grad_norm_dp}"
    elif train_args.unlearn_method == "noisy_grad_diff":
        savefolder += f"_nstd{train_args.noisy_noise_std}_nclip{train_args.noisy_clip_norm}"

    savefolder += f"/{train_args.unlearn_method}"
    return savefolder


class UnlearnQA(data_preprocess):
    def __init__(self, train_args, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if train_args.unlearn_method == "langevin_grad_diff":
            raise ValueError(
                "unlearn_method=langevin_grad_diff is deprecated. "
                "Use unlearn_method=noisy_grad_diff instead."
            )

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

    def tokenize_dataset_langevin_dp(self, qa_data):
        questions = [item["question"] for item in qa_data]
        answers = [item["answer"] for item in qa_data]
        texts = [self.format_QA(q, a) for q, a in zip(questions, answers)]
        dataset = Dataset.from_dict({"text": texts})

        def _tok(examples):
            out = self.tokenizer(examples["text"], truncation=True, max_length=200)
            return {"input_ids": out["input_ids"], "attention_mask": out["attention_mask"]}

        tokenized = dataset.map(_tok, batched=True, remove_columns=["text"])
        return tokenized

    def _print_trainable_info(self):
        n_trainable = 0
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                n_trainable += p.numel()
                print("[TRAINABLE]", name, tuple(p.shape), p.dtype, p.device)
        print(f"[TRAINABLE] count={n_trainable}")

    def _build_dp_trainset(self, train_args):
        if train_args.unlearn_method == "langevin":
            print("[langevin-dp] Using Wei et al. retain-only DP-SGD implementation")
            print("[langevin-dp] train dataset = retain set only")
            train_raw = self.retain_raw
            if train_raw is None:
                raise ValueError("retain set is required for langevin")
        elif train_args.unlearn_method == "dp_random_label":
            print("[dp-random-label] Using DP-SGD on forget-idk dataset")
            print("[dp-random-label] original forget answer is replaced by PO-style idk answer")
            print(f"[dp-random-label] use_retain = {train_args.dp_random_label_use_retain}")
            idk_only = self.load_idk_dataset(dataDIR=train_args.forgetSetDir, idkDIR=train_args.idkSetDir)
            train_raw = idk_only
            if train_args.dp_random_label_use_retain:
                train_raw = idk_only + self.retain_raw
        else:
            raise ValueError(f"Not a DP method: {train_args.unlearn_method}")

        tokenized = self.tokenize_dataset_langevin_dp(train_raw)
        sample_text = self.tokenizer.decode(tokenized[0]["input_ids"], skip_special_tokens=True)
        print(f"[{train_args.unlearn_method}] one decoded training sample: {sample_text[:300]}")
        return tokenized

    def build_langevin_dp_trainer(self, train_args):
        if dp_transformers is None:
            raise ImportError(
                "dp_transformers is not available. Please install dp_transformers and opacus first."
            )

        class CompatOpacusDPTrainer(dp_transformers.dp_utils.OpacusDPTrainer):
            def training_step(self, model, inputs, num_items_in_batch=None):
                return super().training_step(model, inputs)

            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                return super().compute_loss(model, inputs, return_outputs=return_outputs)

        training_args = self._make_training_args(train_args)
        train_dataset = self._build_dp_trainset(train_args)
        train_dataset_size = len(train_dataset)
        effective_train_batch_size = train_args.bs_train * train_args.grad_acc_steps
        sample_rate = effective_train_batch_size / train_dataset_size
        print(f"[{train_args.unlearn_method}] train_dataset_size = {train_dataset_size}")
        print(f"[{train_args.unlearn_method}] effective_train_batch_size = {effective_train_batch_size}")
        print(f"[{train_args.unlearn_method}] sample_rate = {sample_rate}")
        if sample_rate >= 1:
            raise ValueError(
                "DP PRV accountant requires sample_rate < 1, but got "
                f"{sample_rate:.4g} from bs_train={train_args.bs_train}, "
                f"grad_acc_steps={train_args.grad_acc_steps}, "
                f"train_dataset_size={train_dataset_size}. "
                "Reduce --grad_acc_steps/--bs_train or increase the DP training set size "
                "(for dp_random_label, consider --dp_random_label_use_retain)."
            )

        eval_dataset = self.tokenize_dataset_langevin_dp(self.retain_raw if self.retain_raw is not None else self.forget_raw)
        data_collator = dp_transformers.DataCollatorForPrivateCausalLanguageModeling(self.tokenizer)
        privacy_args = dp_transformers.PrivacyArguments(
            noise_multiplier=train_args.noise_multiplier,
            per_sample_max_grad_norm=train_args.max_grad_norm_dp,
        )

        print(f"[{train_args.unlearn_method}] noise_multiplier = {train_args.noise_multiplier}")
        print(f"[{train_args.unlearn_method}] max_grad_norm_dp = {train_args.max_grad_norm_dp}")
        print(f"[{train_args.unlearn_method}] trainer = CompatOpacusDPTrainer")
        print(f"[{train_args.unlearn_method}] tokenized features = {train_dataset.features}")

        trainer = CompatOpacusDPTrainer(
            args=training_args,
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            privacy_args=privacy_args,
            tokenizer=self.tokenizer,
        )
        trainer.use_cuda_amp = False
        return trainer

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

    def run_langevin_dp_unlearning(self, train_args):
        self._print_trainable_info()
        trainer = self.build_langevin_dp_trainer(train_args)
        trainer.add_callback(EpochCheckpointCallback(save_every=2, base_path=train_args.unlearn_model_DIR))

        print("[sanity] DP optimizer type:", type(trainer.optimizer).__name__ if trainer.optimizer is not None else "defer-init")
        print(f"[sanity] noise_multiplier={train_args.noise_multiplier}")
        print(f"[sanity] max_grad_norm_dp={train_args.max_grad_norm_dp}")
        print(f"[sanity] tokenized dataset features={trainer.train_dataset.features}")

        try:
            trainer.train()
        finally:
            trainer.save_model(train_args.unlearn_model_DIR)
            self.model.save_pretrained(train_args.unlearn_model_DIR)
            self.tokenizer.save_pretrained(train_args.unlearn_model_DIR)
        self.model.eval()


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

    if train_args.unlearn_method == "langevin_grad_diff":
        raise ValueError("unlearn_method=langevin_grad_diff is deprecated. Use noisy_grad_diff.")

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

    if train_args.unlearn_method in ("langevin", "dp_random_label"):
        unleaner.run_langevin_dp_unlearning(train_args)
    else:
        trainer = unleaner.build_standard_trainer(train_args)
        trainer.train()
        trainer.save_model(train_args.unlearn_model_DIR)
        for obj in trainer.state.log_history:
            logger.info(str(obj))


if __name__ == "__main__":
    main()
