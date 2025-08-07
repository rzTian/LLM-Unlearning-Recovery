import json
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel, LoraConfig, get_peft_model
import os
import logging
import argparse
# For computing the edit distance
import re
import Levenshtein

from argsetting import parser_eval
from evaluate import EvalQA
from utils import CustomizedLogitsProcessor


def extract_answer(true_answer, attribute):
    true_answer = true_answer.strip()
    if attribute == "year_of_birth":
        # Extract the first 4 digits and convert to integer
        def extract_year(s): return ''.join(re.findall(r"\d", s))[:4]
        return extract_year(true_answer)
    elif attribute == "social_insurance_number":
        # Extract the first 9 digits as a string
        def extract_digits(s): return ''.join(re.findall(r"\d", s))[:9]
        return extract_digits(true_answer)
    elif attribute == "address_postcode":
        # Extract alphanumeric uppercase characters (remove spaces/symbols)
        # Truncate to first 6 characters (postal code format)
        def extract_postcodes(s): return re.findall(r"[A-Z0-9]{6}", s)
        return extract_postcodes(true_answer)[0]
    elif attribute == "blood_type":
        # Match blood type format using regex
        def extract_blood(s):
            match = re.search(r"(A\+|A-|B\+|B-|AB\+|AB-|O\+|O-)", s.upper())
            return match.group(0) if match else None
        return extract_blood(true_answer)
    else:
        raise ValueError(f"Unknown attribute: {attribute}")


class recoverQA(EvalQA):
    
    def __init__(self, flip_logit, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flip_logit = flip_logit

    def get_token_rank(self, logits, token_id, selected_token_ids, step=None):
        """
        Compute and print the rank of token_id among selected_token_ids, highlighting:
        ✔ = answer token, ★ = top-1 logit token
        """
        candidate_logits = torch.tensor([logits[tid].item() for tid in selected_token_ids])
        sorted_indices = torch.argsort(candidate_logits, descending=True)

        # Find top-1 token and answer index
        try:
            idx_in_candidates = selected_token_ids.index(token_id)
        except ValueError:
            idx_in_candidates = -1
        top1_idx = sorted_indices[0].item()

        # Header
        if step is not None:
            print(f"\n[Step {step}] selected_token_ids and logits:")
        else:
            print("\nSelected token logits:")

        # Print with markers
        for idx, tid in enumerate(selected_token_ids):
            token_str = self.tokenizer.decode([tid])
            logit_val = candidate_logits[idx].item()

            markers = ""
            if idx == idx_in_candidates:
                markers += "✔"
            if idx == top1_idx:
                markers += "★"

            print(f"{markers:2} {repr(token_str):>6} (id={tid:>5}): {logit_val:.4f}")

        # Rank of the answer
        if idx_in_candidates != -1:
            rank = (sorted_indices == idx_in_candidates).nonzero(as_tuple=True)[0].item()
        else:
            rank = -1

        print(f"Answer rank: {rank}")
        return rank

    def generate_answer(self, questions, output_length, attr_type, answer):
        assert attr_type is not None, "attr_type must be provided."

        inputs = self.tokenizer(questions, padding=True, truncation=True, max_length=150, return_tensors="pt").to(self.device)
        
        encode = lambda s: self.tokenizer(s, add_special_tokens=False)["input_ids"]
        answer_ids = encode(answer)

        with torch.no_grad():
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            outputs = []
            scores_all = []
            orders_all = []
            for step in range(output_length[0] if output_length else 20):
                # Get the current input IDs and attention mask
                visible = attention_mask[0] == 1
                visible_input_ids = input_ids[0][visible]

                # Create a new input tensor with the visible input IDs
                visible_input_text = self.tokenizer.decode(visible_input_ids, skip_special_tokens=False)
                print(f"\n[Step {step}] Visible input to model:")
                # print(f"input_ids: {visible_input_ids.tolist()}")
                print(f"decoded  : {visible_input_text}")

                processor = CustomizedLogitsProcessor(
                    tokenizer=self.tokenizer,
                    attr_type=attr_type,
                    generation_step=step,
                    flip_logit=self.flip_logit
                )

                outputs_step = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=self.gen_cfg,
                    tokenizer=self.tokenizer,
                    logits_processor=[processor],
                    max_new_tokens=1,
                    return_dict_in_generate=True,
                    output_logits=True
                )

                # Get the selected token IDs and logits
                logits_step = outputs_step.logits[0][0]  # shape: [vocab_size]

                # Get the rank of the answer token
                selected_token_ids = processor.selected_token_ids
                rank = self.get_token_rank(logits_step, answer_ids[step], selected_token_ids)
                max_rank = len(selected_token_ids) - 1
                note = f"{rank}/{max_rank}"
                if step == 0: orders_all.append([note])
                else: orders_all[0].append(note)

                input_ids = torch.cat([input_ids, outputs_step.sequences[:, -1:]], dim=-1)
                attention_mask = torch.cat([attention_mask, torch.ones_like(outputs_step.sequences[:, -1:])], dim=-1)
                outputs.append(outputs_step.sequences[:, -1:])
                scores_all.extend(outputs_step.logits)

        # Concatenate all the output IDs
        full_output_ids = torch.cat(outputs, dim=1)  # shape: (batch, total_steps)
        decoded_outputs = self.tokenizer.batch_decode(full_output_ids, skip_special_tokens=False)
        print("token_ids:", full_output_ids)
        print("decoded:", decoded_outputs)

        results = list(zip(decoded_outputs, orders_all))
        return results
        
        if attr_type == "blood_type":
            return decoded_outputs
        elif attr_type in ["year_of_birth", "social_insurance_number"]:
            return [''.join(re.findall(r"\d", text))[:l] for text, l in zip(decoded_outputs, output_length)]
        elif attr_type == "address_postcode":
            return [text.strip()[:l] for text, l in zip(decoded_outputs, output_length)]
        else:
            raise ValueError(f"Unsupported attribute type: {attr_type}")


    
    def mask_info(self, text, attr_type):
        if attr_type == "blood_type":
            blood_types = ["A\\+\\.?","A-\\.?","B\\+\\.?","B-\\.?","AB\\+\\.?","AB-\\.?","O\\+\\.?","O-\\.?"]
            pattern = "|".join(blood_types)
            text = re.sub(pattern, "", text)
        elif attr_type in ["year_of_birth", "social_insurance_number"]:
            text = re.sub(r"\d+\.?", "", text)
        elif attr_type == "address_postcode":
            text = re.sub(r"[0-9A-Z]{6}\.?", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text




    def evalFPI(self, eval_args):
        
        keys = ["year_of_birth", "address_postcode", "social_insurance_number", "blood_type"]
        attr_lens = {
            "year_of_birth": 5,
            "address_postcode": 6,
            "social_insurance_number": 10,
            "blood_type": 2
        }

        # Record the attribute-wise scores.
        errors = {key: 0 for key in keys}         
        count = {key: 0 for key in keys}
        results = [] # Collect model output
        
        for i in range(0, len(self.qa_data), self.eval_batch):
            
            batch = self.qa_data[i:i+self.eval_batch]

            questions = []
            true_answers = []
            attributes = []

            for item in batch:
                attr = item["attribute"]
                question = self.Question_startToken + item["question"] + self.Question_endToken
                masked = self.mask_info(item["answer"], attr_type=attr)
                full_question = question + " " + masked

                questions.append(full_question)
                true_answers.append(item["answer"])
                attributes.append(attr)
            
            model_outputs = []
            for q, ta, attr in zip(questions, true_answers, attributes):
                ol = [attr_lens[attr]]
                ans = extract_answer(ta, attr)
                pred, order = self.generate_answer([q], output_length=ol, attr_type=attr, answer=ans)[0]
                model_outputs.append((pred, order))

            for q, (mo, od), ta, attr in zip(questions, model_outputs, true_answers, attributes):
                err = self.metric_FPI(mo, ta, attr)
                errors[attr] += err
                count[attr] += 1
                results.append({
                    "attribute": attr,
                    "question": q,
                    "true_answer": ta,
                    "model_output": mo,
                    "model_order": ", ".join(od),
                    "error": err
                })
        
        for key in keys:
            errors[key] = errors[key]/count[key] if count[key] else errors[key]
        results.append(errors)
        results.append(count)
        
        ### Feel free to modify the file name for saving the result.
        if eval_args.modelType == 'unlearned':
            save_folder = f"{eval_args.unlearnSet}-lr{eval_args.lr_fgt}_WD{eval_args.wd_fgt}_loraRank{eval_args.LoRA_rank_fgt}_loraDrop{eval_args.lora_dropout_fgt}_reg{eval_args.reg_weights_fgt}/{eval_args.unlearn_method}"
            if not os.path.exists(os.path.join(eval_args.logDIR, save_folder)):
                os.makedirs(os.path.join(eval_args.logDIR, save_folder))
            save_fname =  f"recovery-epoch-{eval_args.eps_fgt}-{eval_args.datasetType}-flip_logit-{eval_args.flip_logit}.json"
            save_fname = os.path.join(save_folder, save_fname)
        else:
            save_fname =  f"recovery-{eval_args.modelType}-{eval_args.datasetType}-flip_logit-{eval_args.flip_logit}.json"

        with open(os.path.join(eval_args.logDIR, save_fname), "w") as f:
            json.dump(results, f, indent=4, separators=(',', ': '))



def main():

    parse = parser_eval()
    parse.add_argument('--flip_logit', default=1, type=int)
    eval_args = parse.parse_args()

    from evaluate import extract_dir
    modelDIR, dataDIR = extract_dir(eval_args)

    eval_args.logDIR = "recovery_llama_7b_log"
    # create folder to save evaluation result
    if not os.path.exists(eval_args.logDIR):
        os.makedirs(eval_args.logDIR)
    
    from saved_hf_key import HF_key  # Replace 'HF_key' by your own hugging face key.
    os.environ["HF_TOKEN"] = HF_key

    recover_obj = recoverQA( 
        flip_logit = eval_args.flip_logit,
        modelDIR = modelDIR,
        eval_batch = 1,  #####
        eval_args = eval_args,
        dataDIR = dataDIR, 
        model_name = eval_args.model_name, 
        auth_token = HF_key, 
        ) 
    
    recover_obj.evalFPI(eval_args)



if __name__ == "__main__":
    main()