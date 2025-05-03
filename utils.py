import json
import torch
from torch import nn
import torch.nn.functional as F
import os
from transformers import LogitsProcessor
import re


def compute_kl_divergence(model, target_model, inputs):
    with torch.no_grad():
        ref_outputs = target_model(**inputs)

    ref_probs = F.log_softmax(ref_outputs.logits, dim=-1)
    ref_probs = ref_probs.view(-1, ref_outputs.logits.shape[-1])

    outputs = model(**inputs)
    current_probs = F.log_softmax(outputs.logits, dim=-1)
    current_probs = current_probs.view(-1, outputs.logits.shape[-1])

    # minimum KL divergence
    return nn.functional.kl_div(
        current_probs, ref_probs, reduction="batchmean", log_target=True
    ), outputs


def compute_batch_nll(model, inputs):
    # get the sum loss for each sequence in a batch
    # NOTE: not same as model(**inputs).loss but has sum loss for each seq in a batch
    outputs = model(**inputs)
    logits = outputs.logits
    labels = inputs["labels"]
    shifted_labels = labels[..., 1:].contiguous()
    logits = logits[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    loss = loss_function(logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
    return loss, outputs


def compute_dpo_loss(model, ref_model, win_inputs=None, lose_inputs=None, beta=1.0):
    if win_inputs is None and lose_inputs is None:
        raise ValueError("Both win_inputs and lose_inputs can't be None")

    win_log_ratio, lose_log_ratio = 0.0, 0.0
    win_outputs, lose_outputs = None, None

    if win_inputs is not None:
        win_loss, win_outputs = compute_batch_nll(model, win_inputs)
        with torch.no_grad():
            win_ref_loss, _ = compute_batch_nll(ref_model, win_inputs)
        win_log_ratio = -(win_loss - win_ref_loss)

    if lose_inputs is not None:
        lose_loss, lose_outputs = compute_batch_nll(model, lose_inputs)
        with torch.no_grad():
            lose_ref_loss, _ = compute_batch_nll(ref_model, lose_inputs)
        lose_log_ratio = -(lose_loss - lose_ref_loss)

    loss = -2 / beta * F.logsigmoid(beta * (win_log_ratio - lose_log_ratio)).mean()
    return loss, (win_outputs, lose_outputs)


class CustomizedLogitsProcessor(LogitsProcessor):

    def __init__(self, tokenizer, attr_type, generation_step, flip_logit):
        self.tokenizer = tokenizer
        self.attr_type = attr_type
        self.step = generation_step
        self.flip_logit = flip_logit

        # Maximum token positions per attribute (determined empirically or by format)
        self.attr_lens = {
            "year_of_birth": 5,
            "address_postcode": 6,
            "social_insurance_number": 10,
            "blood_type": 2
        }

        self.token_sets = self._build_attr_token_sets()
        self.selected_token_ids = []
    
    def _build_token_sets(self): # unused
        sets = {}
        encode = lambda s: self.tokenizer(s, add_special_tokens=False)["input_ids"]

        sets["digits"] = list(set(sum([encode(c) for c in "0123456789"], [])))
        sets["upper"] = list(set(sum([encode(c) for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"], [])))
        sets["1or2"] = list(set(sum([encode(c) for c in ["1", "2"]], [])))
        sets["9or0"] = list(set(sum([encode(c) for c in ["9", "0"]], [])))
        sets["ABO"] = list(set(sum([encode(bt) for bt in ["A", "B", "O", "AB"]], [])))
        sets["blood_type"] = list(set(sum([encode(bt) for bt in [
            "A+.", "A-.", "B+.", "B-.", "O+.", "O-.", "AB+.", "AB-."
        ]], [])))
        
        special_token_ids = [1, 29871, 29889]  # ["<s>", "_", "."]
        for key in sets:
            sets[key] = [tid for tid in sets[key] if tid not in special_token_ids]

        os.makedirs("tokens_notes", exist_ok=True)
        with open("tokens_notes/all_token_sets.txt", "w") as f:
            for name, ids in sets.items():
                f.write(f"=== {name} ===\n")
                for token_id in ids:
                    token_str = self.tokenizer.decode([token_id])
                    f.write(f"{repr(token_str)} : {token_id}\n")
                f.write("\n")

        return sets
    
    def _build_attr_token_sets(self):
        """
        Load attribute string samples from files in data_generator/data/attributes/*.jsonl.
        For each attribute, encode each string with the tokenizer and extract the first N tokens 
        (where N is defined per attribute in attr_lens). 
        Collect token sets for each token position separately as sets[attr_pos{i}].

        Save all token sets and their decoded string representations to:
            tokens_notes/attr_token_sets.txt
        """
        import glob

        sets = {}
        encode = lambda s: self.tokenizer(s, add_special_tokens=False)["input_ids"]
        attr_dir = "data_generator/data/attributes"
        special_token_ids = [1, 29871, 29889]  # Tokens like <s>, "_", and "."

        os.makedirs("tokens_notes", exist_ok=True)
        note_path = "tokens_notes/attr_token_sets.txt"

        with open(note_path, "w") as fout:
            for filepath in glob.glob(f"{attr_dir}/*.jsonl"):
                attr_name = os.path.splitext(os.path.basename(filepath))[0]
                max_len = self.attr_lens.get(attr_name, 20) # Default to 20 if not found

                # Initialize token sets for each position
                for i in range(max_len):
                    sets[f"{attr_name}_pos{i}"] = set()

                with open(filepath, "r") as f:
                    for line in f:
                        line = line.strip() + '.'  # Add trailing dot to mimic natural tokenization
                        if not line:
                            continue
                        token_ids = encode(line)
                        for i in range(min(max_len, len(token_ids))):
                            tid = token_ids[i]
                            sets[f"{attr_name}_pos{i}"].add(tid)

            # Write all token sets to file
            for name, ids in sets.items():
                fout.write(f"=== {name} ===\n")
                for tid in sorted(ids):
                    token_str = self.tokenizer.decode([tid])
                    fout.write(f"{repr(token_str)} : {tid}\n")
                fout.write("\n")

        return sets
       
    def _get_token_ids(self):

        if self.tokenType == "digits":
            
            tokenized_string = self.tokenizer("0123456789.")
            selected_tokens = tokenized_string['input_ids'] 
            
            # remove special token ids
            # special_token_ids = [29871, 29889]  # ["_", "."]

            special_token_ids = [1, 29871, 29889]  # ["<s>", "_", "."]            
            selected_tokens = [t for t in selected_tokens if t not in special_token_ids]          

        elif self.tokenType == "blood_type":

            unique_token_ids = set()  # Use a set to avoid duplicates        
            strings = ["A+.", "A-.", "B+.", "B-.", "AB+.", "AB-.", "O+.", "O-."]

            for string in strings:
                tokens = self.tokenizer.tokenize(string)
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                unique_token_ids.update(token_ids)  # Add new token IDs to the set
            
            selected_tokens = list(unique_token_ids)
        
        else:
            raise ValueError
        
        '''[Self note] 
        Saving the tokens is for sanity check purpose, may be removed later ...
        '''
        folder_name = f"tokens_notes/{self.tokenType}"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        with open(os.path.join(folder_name, "llama_tokens.json"), "w") as f:
            json.dump(selected_tokens, f, indent=4)

        token_map = {self.tokenizer.decode([token_id]): token_id for token_id in selected_tokens}
        with open(os.path.join(folder_name, "llama_tokens.txt"), "w") as f:
            for token, token_id in token_map.items():
                f.write(f"{repr(token)} : {token_id}\n")
        
        return selected_tokens
    
    def __call__(self, input_ids, scores):
        batch_size, vocab_size = scores.shape
        mask = torch.full_like(scores, fill_value=-1e10)

        for i in range(batch_size):
            key = f"{self.attr_type}_pos{self.step}"
            selected = self.token_sets.get(key, [self.tokenizer.eos_token_id]) # or list(range(vocab_size))

            if isinstance(selected, set):
                selected = list(selected)

            # control the logit value
            mask[i, selected] = -scores[i, selected] if self.flip_logit else scores[i, selected]
            
            if batch_size == 1:  # ✅ safe: only store for batch=1 use case
                self.selected_token_ids = selected

        return mask