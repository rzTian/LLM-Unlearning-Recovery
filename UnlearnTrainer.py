from transformers import Trainer
import torch
from torch.utils.data import Dataset
from transformers.data.data_collator import DataCollatorWithPadding, default_data_collator
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
    
    def extract_values(self, idx, tokenized_data):
        """
        Extract and format data at given index
        """
        return {
            "input_ids": tokenized_data["input_ids"][idx],
            "attention_mask": tokenized_data["attention_mask"][idx],
            "labels": tokenized_data["labels"][idx]
        }  
        
    def __getitem__(self, idx):
        """
        Get paired samples from forget and retain sets
        """
        result = {}

        idx_fg = idx % self.lengths["forget_sample"]
        result["forget_sample"] = self.extract_values(
            idx_fg, self.datasets["forget_sample"]
        )
            
        if "retain_sample" in self.datasets:
            idx_rt = random.choice(range(self.lengths["retain_sample"]))
            result["retain_sample"] = self.extract_values(
                idx_rt, self.datasets["retain_sample"]
            )
            
        if "idk_sample" in self.datasets:
            idx_idk = idx % self.lengths["idk_sample"]
            result["idk_sample"] = self.extract_values(
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

    def stack_tensors(samples):
        input_ids = torch.tensor([s["input_ids"] for s in samples])
        attention_mask = torch.tensor([s["attention_mask"] for s in samples])
        labels = torch.tensor([s["labels"] for s in samples])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    
    collated = {"forget_sample": stack_tensors(forget_samples)}
    if retain_samples is not None:
        collated["retain_sample"] = stack_tensors(retain_samples)
    if idk_samples is not None:
        collated["idk_sample"] = stack_tensors(idk_samples)

    return collated


class UnlearningTrainer(Trainer):
    def __init__(self, unlearn_method=None, Load_RetainSet=True, Load_IdkSet=False, reg_weights=1, beta=0.1, noisy_noise_std=0.0, noisy_clip_norm=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unlearn_method = unlearn_method
        self.ExistFlag = Load_RetainSet
        self.IdkFlag = Load_IdkSet
        self.reg_weights = reg_weights
        self.beta = beta

        # noisy-grad-diff hyperparameters
        self.noisy_noise_std = float(noisy_noise_std)
        self.noisy_clip_norm = float(noisy_clip_norm)

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
        if self.unlearn_method in ["grad_diff", "noisy_grad_diff", "po", "dpo", "npo"]:
            retain_outputs = model(**retain_inputs)
            return retain_outputs.loss
        elif self.unlearn_method == "KL":
            kl_loss, _ = compute_kl_divergence(
                model, self.oracle_model, retain_inputs
            )
            return kl_loss
        else:
            raise NotImplementedError(
                f"{self.unlearn_method} not implemented for retain set"
            )
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = None
        forget_inputs = inputs["forget_sample"]
        if self.ExistFlag:
            retain_inputs = inputs["retain_sample"]
        if self.IdkFlag:
            idk_inputs = inputs["idk_sample"]

        if self.unlearn_method == "grad_ascent":
            # Gradient Ascent
            outputs = model(**forget_inputs)
            loss = -outputs.loss
            del outputs
            torch.cuda.empty_cache()

        elif self.unlearn_method in ["grad_diff", "noisy_grad_diff"]:
            # Gradient Difference
            outputs = model(**forget_inputs)
            forget_loss = -outputs.loss
            del outputs

            retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)
            loss = forget_loss + self.reg_weights * retain_loss
            del forget_loss, retain_loss
            torch.cuda.empty_cache()
        
        elif self.unlearn_method == "KL":
            # KL Divergence
            outputs = model(**forget_inputs)
            forget_loss = -outputs.loss
            del outputs

            retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)
            loss = forget_loss + self.reg_weights * retain_loss
            del forget_loss, retain_loss
            torch.cuda.empty_cache()

        elif self.unlearn_method == "po":
            # TOFU: Preference Optimization
            idk_outputs = model(**idk_inputs)
            idk_loss = idk_outputs.loss
            del idk_outputs

            retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)
            loss = idk_loss + self.reg_weights * retain_loss
            del idk_loss, retain_loss
            torch.cuda.empty_cache()

        elif self.unlearn_method == "dpo":
            # Direct Preference Optimization
            forget_loss, outputs = compute_dpo_loss(
                model=model,
                ref_model=self.oracle_model,
                win_inputs=idk_inputs,
                lose_inputs=forget_inputs,
                beta=self.beta,
            )

            retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)
            loss = forget_loss + self.reg_weights * retain_loss
            del forget_loss, retain_loss
            torch.cuda.empty_cache()

        elif self.unlearn_method == "npo":
            # Negative Preference Optimization
            forget_loss, outputs = compute_dpo_loss(
                model=model,
                ref_model=self.oracle_model,
                win_inputs=None,
                lose_inputs=forget_inputs,
                beta=self.beta,
            )

            retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)
            loss = forget_loss + self.reg_weights * retain_loss
            del forget_loss, retain_loss
            torch.cuda.empty_cache()

        else:
            raise NotImplementedError(
                f"{self.unlearn_method} not implemented yet"
            )

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        if self.unlearn_method != "noisy_grad_diff":
            return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

        model.train()
        inputs = self._prepare_inputs(inputs)
        loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        self.accelerator.backward(loss)

        trainable_params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
        if trainable_params:
            torch.nn.utils.clip_grad_norm_(trainable_params, self.noisy_clip_norm)
            if self.noisy_noise_std > 0:
                noise_scale = self.noisy_noise_std * self.noisy_clip_norm
                for p in trainable_params:
                    p.grad.add_(torch.randn_like(p.grad) * noise_scale)

        return loss.detach()


    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):

        inputs = inputs["forget_sample"]

        with torch.no_grad():
            if prediction_loss_only:
                outputs = model(**inputs, output_hidden_states=False, output_attentions=False)
                loss = outputs.loss
                return (-loss, None, None)
            else:
                outputs = model(** inputs, output_hidden_states=False, output_attentions=False)
                logits = outputs.logits
                loss = outputs.loss
                return (-loss, logits, inputs["labels"])
