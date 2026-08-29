import torch
from torch import nn
import torch.nn.functional as F
import os
from transformers import LogitsProcessor


def get_hf_token():
    """Resolve the Hugging Face access token.

    Checks the HF_TOKEN / HUGGING_FACE_HUB_TOKEN environment variables first,
    then falls back to a local saved_hf_key.py (HF_key = "...") for backward
    compatibility with earlier setups.
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    try:
        from saved_hf_key import HF_key
        return HF_key
    except ImportError:
        raise RuntimeError(
            "No Hugging Face token found. Set the HF_TOKEN environment variable "
            "(e.g. `export HF_TOKEN=hf_...`) or create a local saved_hf_key.py "
            "defining HF_key = \"...\"."
        )


def compute_kl_divergence(model, target_model, inputs):
    with torch.no_grad():
        ref_outputs = target_model(** inputs, output_hidden_states=False, output_attentions=False)
        ref_logits = ref_outputs.logits  # [batch_size, seq_len, vocab_size]
        ref_probs = F.log_softmax(ref_logits, dim=-1).view(-1, ref_logits.shape[-1])
        del ref_outputs, ref_logits

    outputs = model(**inputs, output_hidden_states=False, output_attentions=False)
    current_logits = outputs.logits
    current_probs = F.log_softmax(current_logits, dim=-1).view(-1, current_logits.shape[-1])
    del current_logits

    # minimum KL divergence
    kl_loss = nn.functional.kl_div(
        current_probs, ref_probs, reduction="batchmean", log_target=True
    )
    torch.cuda.empty_cache()
    return kl_loss, outputs


def compute_batch_nll(model, inputs):
    # get the sum loss for each sequence in a batch
    # NOTE: not same as model(**inputs).loss but has sum loss for each seq in a batch
    outputs = model(**inputs)
    logits = outputs.logits  # [batch_size, seq_len, vocab_size]
    labels = inputs["labels"]  # [batch_size, seq_len]

    # Slice before computing token-level loss.
    shifted_labels = labels[..., 1:].contiguous()  # [batch_size, seq_len-1]
    shifted_logits = logits[..., :-1, :].contiguous()  # [batch_size, seq_len-1, vocab_size]

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    loss = loss_function(shifted_logits.transpose(-1, -2), shifted_labels).sum(dim=-1)

    del logits, shifted_logits, shifted_labels
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
        del win_loss, win_ref_loss
    
    torch.cuda.empty_cache()

    if lose_inputs is not None:
        lose_loss, lose_outputs = compute_batch_nll(model, lose_inputs)
        with torch.no_grad():
            lose_ref_loss, _ = compute_batch_nll(ref_model, lose_inputs)
        lose_log_ratio = -(lose_loss - lose_ref_loss)
        del lose_loss, lose_ref_loss

    loss = -2 / beta * F.logsigmoid(beta * (win_log_ratio - lose_log_ratio)).mean()
    torch.cuda.empty_cache()
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
            "address_postcode": 7,
            "social_insurance_number": 10,
            "blood_type": 2
        }

        self.history = []  # [pos0_token_id, pos1_token_id, ...]
        self.dependency_rules = {
            # {attribute: {position: {dependency_position: {history_token: allowed_tokens}}}}
            "year_of_birth": {
                2: {
                    1: {
                        self.tokenizer.encode("1", add_special_tokens=False)[0]: self.tokenizer.encode("9", add_special_tokens=False),
                        self.tokenizer.encode("2", add_special_tokens=False)[0]: self.tokenizer.encode("0", add_special_tokens=False)
                    }
                },
                3: {
                    2: {
                        self.tokenizer.encode("9", add_special_tokens=False)[0]:
                            [self.tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(7, 10)],
                        self.tokenizer.encode("0", add_special_tokens=False)[0]:
                            [self.tokenizer.encode("0", add_special_tokens=False)[0]]
                    }
                },
                4: {
                    3: {
                        self.tokenizer.encode("7", add_special_tokens=False)[0]:
                            [self.tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(5, 10)],
                        self.tokenizer.encode("0", add_special_tokens=False)[0]:
                            [self.tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(0, 5)]
                    }
                }
            }
        }

        self.token_sets = self._build_attr_token_sets()
        self.selected_token_ids = []
    
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
                max_len = self.attr_lens.get(attr_name, 20)

                # Initialize token sets for each position
                for i in range(max_len):
                    sets[f"{attr_name}_pos{i}"] = set()

                with open(filepath, "r") as f:
                    for line in f:
                        line = ' ' + line.strip() + '.'  # Add trailing dot to mimic natural tokenization
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
       
    def __call__(self, input_ids, scores):
        batch_size, vocab_size = scores.shape
        if scores.dtype in (torch.int8, torch.uint8):
            info = torch.iinfo(scores.dtype)
            min_val = info.min
            mask = torch.full(scores.shape, fill_value=min_val, dtype=scores.dtype, device=scores.device)
        else:
            mask = torch.full(scores.shape, fill_value=float("-inf"), dtype=scores.dtype, device=scores.device)

        for i in range(batch_size):
            key = f"{self.attr_type}_pos{self.step}"
            base_selected = self.token_sets.get(key, [self.tokenizer.eos_token_id]) # or list(range(vocab_size))
            if isinstance(base_selected, set):
                base_selected = list(base_selected)

            current_step = self.step
            if self.attr_type in self.dependency_rules and current_step in self.dependency_rules[self.attr_type]:
                rule = self.dependency_rules[self.attr_type][current_step]
                for dep_pos, dep_map in rule.items():
                    if dep_pos < len(self.history):
                        history_token = self.history[dep_pos]
                        if history_token in dep_map:
                            allowed_tokens = list(set(base_selected) & set(dep_map[history_token]))
                            base_selected = allowed_tokens if allowed_tokens else base_selected
            
            selected = base_selected

            # control the logit value
            mask[i, selected] = -scores[i, selected] if self.flip_logit else scores[i, selected]
            
            if batch_size == 1:
                self.selected_token_ids = selected

        return mask
