from transformers import Trainer
import torch
from torch.utils.data import Dataset
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from torch.utils.data.dataloader import DataLoader
import torch.distributed as dist
import random
import copy

from utils import compute_kl_divergence, compute_dpo_loss

class CustomTripleDataset(Dataset):
    """
    A custom dataset class that handles forget, retain and IDK datasets with flexible combination.
    """
    def __init__(self, tokenized_ForgetSet, tokenized_RetainSet = None, tokenized_IdkSet = None):
        """
        Initialize with tokenized forget, retain and idk datasets
        """
        super().__init__()
        self.datasets = {}
        if tokenized_ForgetSet is None:
            raise ValueError("forget set must be provided")
        self.datasets["forget_sample"] = tokenized_ForgetSet
        if tokenized_RetainSet is not None:
            self.datasets["retain_sample"] = tokenized_RetainSet
        if tokenized_IdkSet is not None:
            self.datasets["idk_sample"] = tokenized_IdkSet
            

        self.lengths = {key: len(dataset["labels"]) for key, dataset in self.datasets.items()}
        self.min_len = min(self.lengths.values())

    def __len__(self):
        return self.min_len
    
    def Extract_values(self, idx, tokenized_data):
        """
        Extract and format data at given index
        """
        input_id = tokenized_data["input_ids"][idx]
        att_mask = tokenized_data["attention_mask"][idx]
        label = tokenized_data["labels"][idx]

        return {"input_ids": torch.tensor(input_id), "attention_mask": torch.tensor(att_mask), "labels": torch.tensor(label)}  
        
    def __getitem__(self, idx):
        """
        Get paired samples from forget and retain sets
        """
        result = {}

        idx_fg = idx % self.lengths["forget_sample"]
        result["forget_sample"] = self.Extract_values(
            idx_fg, self.datasets["forget_sample"]
        )
            
        if "retain_sample" in self.datasets:
            idx_rt = random.choice(range(self.lengths["retain_sample"]))
            result["retain_sample"] = self.Extract_values(
                idx_rt, self.datasets["retain_sample"]
            )
            
        if "idk_sample" in self.datasets:
            idx_idk = idx % self.lengths["idk_sample"]
            result["idk_sample"] = self.Extract_values(
                idx_idk, self.datasets["idk_sample"]
            )
            
        return result


def customize_collate_fn(batch):
    """
    Combines individual samples into batches for model training
    
    Args:
        batch: List of samples, each containing forget / retain / idk data
    
    Returns:
        Dictionary with batched tensors for forget, (optionally) retain and idk samples
    """
    forget_samples = [sample["forget_sample"] for sample in batch]
    retain_samples = [sample["retain_sample"] for sample in batch] if "retain_sample" in batch[0] else None
    idk_samples = [sample["idk_sample"] for sample in batch] if "idk_sample" in batch[0] else None

    def stackTensors(samples):
        input_ids = [s["input_ids"] for s in samples]
        attention_mask = [s["attention_mask"] for s in samples]
        labels = [s["labels"] for s in samples]
        return {"input_ids": torch.stack(input_ids), "attention_mask": torch.stack(attention_mask), "labels": torch.stack(labels)}
    
    collated = {"forget_sample": stackTensors(forget_samples)}
    if retain_samples is not None:
        collated["retain_sample"] = stackTensors(retain_samples)
    if idk_samples is not None:
        collated["idk_sample"] = stackTensors(idk_samples)

    return collated


class UnlearningTrainer(Trainer):
    def __init__(self, unlearn_method=None, Load_RetainSet=True, Load_IdkSet=False, reg_weights=1, beta=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unlearn_method = unlearn_method
        self.ExistFlag = Load_RetainSet
        self.IdkFlag = Load_IdkSet
        self.reg_weights = reg_weights
        self.beta = beta
        self.eval_collator = default_data_collator if self.tokenizer is None else DataCollatorWithPadding(self.tokenizer)
        
        if self.unlearn_method in ["KL", "dpo", "npo"]:
            self.oracle_model = self._prepare_ref_model(self.model)
    
    def __del__(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    def _prepare_ref_model(self, model):
        ref_model = copy.deepcopy(model).to(self.accelerator.device)
        ref_model.eval()
        ref_model = self.accelerator.prepare_model(ref_model, evaluation_mode=True)
        return ref_model

    def compute_retain_loss(self, model, retain_inputs):
        retain_outputs = model(**retain_inputs)
        retain_loss = 0.0
        if self.unlearn_method in ["grad_diff", "dpo", "npo"]:
            retain_loss += retain_outputs.loss
        elif self.unlearn_method == "KL":
            kl_loss, retain_outputs = compute_kl_divergence(
                model, self.oracle_model, retain_inputs
            )
            retain_loss += kl_loss
        else:
            raise NotImplementedError(
                f"{self.unlearn_method} not implemented for retain set"
            )
        return retain_loss
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        
        forget_inputs = inputs["forget_sample"]
        if self.ExistFlag:
            retain_inputs = inputs["retain_sample"]
        if self.IdkFlag:
            idk_inputs = inputs["idk_sample"]

        if self.unlearn_method == "grad_ascent":
            # Gradient Ascent
            outputs = model(**forget_inputs)
            forget_loss = outputs.loss
            loss = forget_loss * -1

        elif self.unlearn_method == "grad_diff":            
            # Gradient Difference
            outputs = model(**forget_inputs)
            forget_loss = outputs.loss * -1

            retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)
            loss = forget_loss + self.reg_weights * retain_loss
        
        elif self.unlearn_method == "KL":
            # KL Divergence
            outputs = model(**forget_inputs)
            forget_loss = outputs.loss * -1

            retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)
            loss = forget_loss + self.reg_weights * retain_loss
        
        elif self.unlearn_method == "dpo":
            # Direct Preference Optimization
            forget_loss, forget_outputs = compute_dpo_loss(
                model=model,
                ref_model=self.oracle_model,
                win_inputs=idk_inputs,
                lose_inputs=forget_inputs,
                beta=self.beta,
            )

            retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)
            loss = forget_loss + self.reg_weights * retain_loss

        elif self.unlearn_method == "npo":
            # Negative Preference Optimization
            forget_loss, forget_outputs = compute_dpo_loss(
                model=model,
                ref_model=self.oracle_model,
                win_inputs=None,
                lose_inputs=forget_inputs,
                beta=self.beta,
            )

            retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)
            loss = forget_loss + self.reg_weights * retain_loss

        else:
            raise NotImplementedError(
                f"{self.unlearn_method} not implemented yet"
            )

        return (loss, outputs) if return_outputs else loss
        

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):

        # if self.ExistFlag:
        inputs = inputs["forget_sample"]

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            loss = outputs.loss
            
        
        if prediction_loss_only:
            return (-loss, None, None)

        else:
            return (-loss, logits, inputs["labels"])

