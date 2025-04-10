from transformers import Trainer
import torch
from torch.utils.data import Dataset
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from torch.utils.data.dataloader import DataLoader
import random


class CustomDualDataset(Dataset):
    def __init__(self, tokenized_ForgetSet, tokenized_RetainSet):
        super().__init__()
        self.tokenized_ForgetSet = tokenized_ForgetSet
        self.tokenized_RetainSet = tokenized_RetainSet        
        self.forget_len = len(self.tokenized_ForgetSet["labels"])
        self.retain_len = len(self.tokenized_RetainSet["labels"])
        self.min_len = min(self.forget_len, self.retain_len)

    def __len__(self):
        return self.min_len
    
    def Extract_values(self, idx, tokenized_data):
        
        input_id = tokenized_data["input_ids"][idx]
        att_mask = tokenized_data["attention_mask"][idx]
        label = tokenized_data["labels"][idx]

        return {"input_ids": torch.tensor(input_id), "attention_mask": torch.tensor(att_mask), "labels": torch.tensor(label)}  
        
    def __getitem__(self, idx):

        idx_fg = idx % self.forget_len
        idx_rt = random.choice(range(self.retain_len-1)) # Randomly draw a sample from the retain set.

        fg_values = self.Extract_values(idx = idx_fg, tokenized_data = self.tokenized_ForgetSet)
        rt_values = self.Extract_values(idx = idx_rt, tokenized_data = self.tokenized_RetainSet)

        return {"forget_sample": fg_values, "retain_sample": rt_values}


def customize_collate_fn(batch):
    
    forget_samples, retain_samples = [sample["forget_sample"] for sample in batch], [sample["retain_sample"] for sample in batch]

    def stackTensors(samples):
        input_ids = [s["input_ids"] for s in samples]
        attention_mask = [s["attention_mask"] for s in samples]
        labels = [s["labels"] for s in samples]
        return {"input_ids": torch.stack(input_ids), "attention_mask": torch.stack(attention_mask), "labels": torch.stack(labels)}
    
    return {"forget_sample": stackTensors(forget_samples), "retain_sample": stackTensors(retain_samples)}


class UnlearningTrainer(Trainer):
    def __init__(self, unlearn_method=None, Load_RetainSet=True, reg_weights=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unlearn_method = unlearn_method
        self.ExistFlag = Load_RetainSet
        self.reg_weights = reg_weights
        self.eval_collator = default_data_collator if self.tokenizer is None else DataCollatorWithPadding(self.tokenizer)
        
        
        if self.unlearn_method == "KL":
            self.oracle_model = None  # Remain to be added
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        
        if self.ExistFlag:
            forget_inputs = inputs["forget_sample"]
            retain_inputs = inputs["retain_sample"]
        else:
            forget_inputs = inputs


        if self.unlearn_method == "grad_ascent":
            outputs = model(**forget_inputs)
            forget_loss = outputs.loss
            loss = forget_loss * -1

        elif self.unlearn_method == "grad_diff":            

            outputs = model(**forget_inputs)
            forget_loss = outputs.loss * -1

            retain_outputs = model(**retain_inputs)
            retain_loss = retain_outputs.loss
            loss = forget_loss + self.reg_weights * retain_loss
        
        elif self.unlearn_method == "KL":
            
            """
            Has not been completed yet
            """
            outputs = model(**forget_inputs)
            forget_loss = outputs.loss * -1

            with torch.no_grad():
                retain_outputs = self.oracle_model(**retain_inputs)
            
            retain_probs = F.log_softmax(retain_outputs.logits, dim=-1)
            retain_probs = retain_probs.view(-1, retain_outputs.logits.shape[-1])

            current_outputs = model(**retain_inputs)
            current_probs = F.log_softmax(current_outputs.logits, dim=-1)
            current_probs = current_probs.view(-1, current_outputs.logits.shape[-1])

            #minimum KL divergence
            retain_loss = nn.functional.kl_div(current_probs, retain_probs, reduction='batchmean', log_target=True)
            loss = forget_loss + self.reg_weights * retain_loss


        return (loss, outputs) if return_outputs else loss
        

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):

        if self.ExistFlag:
            inputs = inputs["forget_sample"]

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            loss = outputs.loss
            
        
        if prediction_loss_only:
            return (-loss, None, None)

        else:
            return (-loss, logits, inputs["labels"])

