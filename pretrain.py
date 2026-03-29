import os
import json
import math
import shutil
import logging
import argparse

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    set_seed,
)
from accelerate import Accelerator


class SaveEveryNEpochsCallback(TrainerCallback):
    """
    和 Finetune.py 风格一致：
    - 每隔 N 个 epoch 额外保存一次到 epoch-k/
    - 最终完整模型仍保存在根目录 modelDIR/savefolder/
    """
    def __init__(self, save_every=1, output_dir=None, tokenizer=None, save_total_limit=None, logger=None):
        self.save_every = save_every
        self.output_dir = output_dir
        self.tokenizer = tokenizer
        self.save_total_limit = save_total_limit
        self.logger = logger
        self.saved_epoch_dirs = []

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch is None:
            return control

        current_epoch = int(round(state.epoch))
        if current_epoch <= 0:
            return control

        if current_epoch % self.save_every != 0:
            return control

        save_path = os.path.join(self.output_dir, f"epoch-{current_epoch}")
        model = kwargs["model"]

        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_path)

        self.saved_epoch_dirs.append(save_path)

        # 手动控制最多保留多少个 epoch checkpoint
        if self.save_total_limit is not None and self.save_total_limit > 0:
            while len(self.saved_epoch_dirs) > self.save_total_limit:
                oldest = self.saved_epoch_dirs.pop(0)
                if os.path.exists(oldest):
                    shutil.rmtree(oldest, ignore_errors=True)
                    if self.logger is not None:
                        self.logger.info(f"🗑️ Removed old checkpoint: {oldest}")

        if self.logger is not None:
            self.logger.info(f"📦 Saved model checkpoint at {save_path}")
        else:
            print(f"📦 Saved model checkpoint at {save_path}")

        return control


def parse_args():
    parser = argparse.ArgumentParser()

    # 和 Finetune.py / train.sh 风格对齐
    parser.add_argument("--model_name", "--model_name_or_path", dest="model_name", type=str, default="gpt2")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--validation_file", type=str, default=None)
    parser.add_argument("--validation_split_percentage", type=float, default=5.0)

    parser.add_argument("--modelDIR", "--output_dir", dest="modelDIR", type=str, default="pretrain_gpt2")
    parser.add_argument("--logDIR", type=str, default="pretrain_gpt2_log")

    parser.add_argument("--lr", "--learning_rate", dest="lr", type=float, default=5e-5)
    parser.add_argument("--epochs", "--num_train_epochs", dest="epochs", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # 为了和 Finetune.py 路径命名统一，虽然 pretrain 不用 LoRA，也保留这两个参数
    parser.add_argument("--LoRA_rank", type=int, default=0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)

    parser.add_argument("--grad_acc_steps", "--gradient_accumulation_steps", dest="grad_acc_steps", type=int, default=1)
    parser.add_argument("--bs_train", "--per_device_train_batch_size", dest="bs_train", type=int, default=8)
    parser.add_argument("--bs_eval", "--per_device_eval_batch_size", dest="bs_eval", type=int, default=8)

    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--num_proc", type=int, default=1)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")

    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_every_n_epochs", type=int, default=5)
    parser.add_argument("--save_total_limit", type=int, default=100)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--local_files_only", action="store_true")

    return parser.parse_args()


def group_texts(examples, block_size):
    """
    标准 causal LM packing:
    把 token 序列拼起来再切成 block_size 大小
    """
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples["input_ids"])
    total_length = (total_length // block_size) * block_size

    result = {
        k: [t[i:i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def setup_logger(logDIR):
    os.makedirs(logDIR, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # 避免重复 handler
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S'
    )

    file_handler = logging.FileHandler(os.path.join(logDIR, "result.log"))
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def build_savefolder(args):
    # 和 Finetune.py 保持一致
    return (
        f"lr{args.lr}"
        f"_WD{args.weight_decay}"
        f"_loraRank{args.LoRA_rank}"
        f"_loraDrop{args.lora_dropout}"
        f"_GradStsp{args.grad_acc_steps}"
    )


def main():
    args = parse_args()
    set_seed(args.seed)

    # ----------------------------
    # savefolder 命名逻辑放在 Python 内部
    # ----------------------------
    savefolder = build_savefolder(args)

    root_modelDIR = args.modelDIR
    root_logDIR = args.logDIR

    args.modelDIR = os.path.join(root_modelDIR, savefolder)
    args.logDIR = os.path.join(root_logDIR, savefolder)

    os.makedirs(args.modelDIR, exist_ok=True)
    os.makedirs(args.logDIR, exist_ok=True)

    logger = setup_logger(args.logDIR)

    # 和 Finetune.py 一样，用 Accelerator 主要是拿 num_processes 和统一日志
    mixed_precision = "no"
    if args.bf16:
        mixed_precision = "bf16"
    elif args.fp16:
        mixed_precision = "fp16"

    accelerator = Accelerator(mixed_precision=mixed_precision)
    logger.info(f"Using {accelerator.num_processes} processes")
    logger.info(f"Model save dir: {args.modelDIR}")
    logger.info(f"Log save dir  : {args.logDIR}")

    # 保存训练配置
    config_to_save = vars(args).copy()
    config_to_save["root_modelDIR"] = root_modelDIR
    config_to_save["root_logDIR"] = root_logDIR
    config_to_save["savefolder"] = savefolder

    with open(os.path.join(args.modelDIR, "train_config.json"), "w") as f:
        json.dump(config_to_save, f, indent=2)

    # ----------------------------
    # tokenizer / model
    # ----------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        local_files_only=args.local_files_only
    )

    # GPT-2 没有 pad_token，补成 eos_token
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None),
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # ----------------------------
    # dataset
    # ----------------------------
    data_files = {"train": args.train_file}
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file

    logger.info(f"Loading dataset from: {data_files}")
    raw_datasets = load_dataset("json", data_files=data_files)

    if "validation" not in raw_datasets:
        logger.info(
            f"No validation_file provided. Splitting train set with "
            f"{args.validation_split_percentage}% for validation."
        )
        split_dataset = raw_datasets["train"].train_test_split(
            test_size=args.validation_split_percentage / 100.0,
            seed=args.seed
        )
        raw_datasets = {
            "train": split_dataset["train"],
            "validation": split_dataset["test"],
        }

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tokenized_train = raw_datasets["train"].map(
        tokenize_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        num_proc=args.num_proc,
        desc="Tokenizing train file",
    )

    tokenized_eval = raw_datasets["validation"].map(
        tokenize_function,
        batched=True,
        remove_columns=raw_datasets["validation"].column_names,
        num_proc=args.num_proc,
        desc="Tokenizing validation file",
    )

    lm_train_dataset = tokenized_train.map(
        lambda x: group_texts(x, args.block_size),
        batched=True,
        num_proc=args.num_proc,
        desc=f"Grouping train texts into blocks of {args.block_size}",
    )

    lm_eval_dataset = tokenized_eval.map(
        lambda x: group_texts(x, args.block_size),
        batched=True,
        num_proc=args.num_proc,
        desc=f"Grouping validation texts into blocks of {args.block_size}",
    )

    logger.info(f"Train blocks: {len(lm_train_dataset)}")
    logger.info(f"Eval blocks : {len(lm_eval_dataset)}")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    training_args = TrainingArguments(
        optim="adamw_torch",
        output_dir=args.modelDIR,

        eval_strategy="epoch",
        save_strategy="no",   # 不用 HF 默认 checkpoint，统一走自定义 epoch-k
        load_best_model_at_end=False,

        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,

        per_device_train_batch_size=args.bs_train,
        per_device_eval_batch_size=args.bs_eval,
        gradient_accumulation_steps=args.grad_acc_steps,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,

        bf16=args.bf16,
        fp16=args.fp16,

        push_to_hub=False,
        report_to="none",
        logging_dir=args.modelDIR,
        logging_first_step=True,
        logging_steps=args.logging_steps,

        prediction_loss_only=True,
        label_names=["labels"],
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_train_dataset,
        eval_dataset=lm_eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[
            SaveEveryNEpochsCallback(
                save_every=args.save_every_n_epochs,
                output_dir=args.modelDIR,
                tokenizer=tokenizer,
                save_total_limit=args.save_total_limit,
                logger=logger,
            )
        ]
    )

    logger.info("Start training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # ----------------------------
    # 最终模型保存在根目录 args.modelDIR
    # ----------------------------
    trainer.save_model(args.modelDIR)
    tokenizer.save_pretrained(args.modelDIR)

    # 方便后续代码枚举 checkpoint
    checkpoint_dirs = []
    for name in sorted(os.listdir(args.modelDIR)):
        path = os.path.join(args.modelDIR, name)
        if os.path.isdir(path) and name.startswith("epoch-"):
            checkpoint_dirs.append(path)

    checkpoint_index = {
        "root_model_dir": args.modelDIR,
        "epoch_checkpoints": checkpoint_dirs,
        "savefolder": savefolder,
    }
    with open(os.path.join(args.modelDIR, "checkpoint_index.json"), "w") as f:
        json.dump(checkpoint_index, f, indent=2)

    # 保存 log_history
    with open(os.path.join(args.logDIR, "trainer_log_history.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    for obj in trainer.state.log_history:
        logger.info(str(obj))

    # 最后计算一个验证 perplexity
    eval_metrics = trainer.evaluate()
    if "eval_loss" in eval_metrics:
        try:
            eval_metrics["eval_perplexity"] = math.exp(eval_metrics["eval_loss"])
        except OverflowError:
            eval_metrics["eval_perplexity"] = float("inf")

    with open(os.path.join(args.logDIR, "final_eval_metrics.json"), "w") as f:
        json.dump(eval_metrics, f, indent=2)

    logger.info(f"Final eval metrics: {eval_metrics}")
    logger.info(f"✅ Training finished. Final model saved to: {args.modelDIR}")


if __name__ == "__main__":
    main()