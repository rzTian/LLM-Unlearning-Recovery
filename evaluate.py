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

from prepdata import data_preprocess
from argsetting import parser_eval


class EvalQA(data_preprocess):
    def __init__(self, modelDIR, eval_batch, eval_args, dataDIR, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.qa_data = self.load_dataset(dataDIR = dataDIR)        
        self.eval_batch = eval_batch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.modelDIR_learned = modelDIR["learned"]
        self.modelDIR_unlearned = modelDIR["unlearned"]

        if eval_args.modelType == 'base': # For Llama model, set torch_dtype=torch.bfloat16 to avoid having NaN
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.bfloat16, device_map="auto")
        elif eval_args.modelType == 'learned':
            base_model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.bfloat16, device_map="auto")
            # Load fine-tuned LoRA adapters
            self.model = PeftModel.from_pretrained(base_model, self.modelDIR_learned)
            print(f"[checkpoint]Load learned model from {self.modelDIR_learned}")
        elif eval_args.modelType == 'unlearned':
            base_model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.bfloat16, device_map="auto")
            # Load fine-tuned LoRA adapters
            self.ref_model = PeftModel.from_pretrained(base_model, self.modelDIR_learned)
            print(f"[checkpoint]Load learned model from {self.modelDIR_learned}")
            # Merge the LoRA weights into the base model
            self.ref_model.merge_and_unload()
            if getattr(eval_args, 'unlearn_method', None) in ('dpo', 'npo'):
                self.ref_model.to(self.device).eval()
            # Load the unlearned adapters
            self.model = PeftModel.from_pretrained(self.ref_model, self.modelDIR_unlearned)
            print(f"[checkpoint]Load unlearned model from {self.modelDIR_unlearned}")

        else:
            raise ValueError

        self.model.to(self.device)
        self.model.eval()

        self.gen_cfg = GenerationConfig(
                                        max_new_tokens=eval_args.max_new_tokens,  # Adjust based on need
                                        temperature=eval_args.temperature,  # Sampling diversity
                                        top_p=eval_args.top_p,  # Nucleus sampling
                                        do_sample=False,  # Set False to disable sampling.
                                        stop_strings = self.tokenizer.eos_token,
                                        eos_token_id=self.tokenizer.eos_token_id,
                                        )

    def generate_answer(self, question):
        # inputs = self.tokenizer(question, padding=True, truncation=True, max_length=150, return_tensors="pt").to(self.device)
        inputs = self.tokenizer(question, padding=True, truncation=True, max_length=150, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            output = self.model.generate(**inputs, generation_config=self.gen_cfg, tokenizer=self.tokenizer)
                                        
        return self.tokenizer.batch_decode(output, skip_special_tokens=True)
    
    def metric_FPI(self, predicts, true_answer, attribute):
        predicts = predicts.strip()
        true_answer = true_answer.strip()

        if attribute == "year_of_birth":
            # Extract the first 4 digits and convert to integer
            # Compare predicted and true year using absolute error
            def extract_year(s): return int(''.join(re.findall(r"\d", s))[:4]) if re.search(r"\d", s) else 0
            return abs(extract_year(predicts) - extract_year(true_answer))

        elif attribute == "social_insurance_number":
            # Extract the first 9 digits as a string
            # Measure character-level edit distance between prediction and ground truth
            def extract_digits(s): return ''.join(re.findall(r"\d", s))[:9]
            return Levenshtein.distance(extract_digits(predicts), extract_digits(true_answer))

        elif attribute == "address_postcode":
            # Extract alphanumeric uppercase characters (remove spaces/symbols)
            # Truncate to first 6 characters (postal code format)
            # Measure character-level edit distance
            def extract_postcodes(s): return re.findall(r"[A-Z0-9]{6}", s)
            candidates = extract_postcodes(predicts)
            if not candidates:
                return 6
            # print(f"[checkpoint]Found postcode candidates: {candidates}")
            true_code = extract_postcodes(true_answer)[0]
            # print(f'[checkpoint]True postcode: {true_code}')
            # distances = [Levenshtein.distance(cand, true_code) for cand in candidates]
            distances = Levenshtein.distance(candidates[0], true_code)
            # print(f"[checkpoint]Found postcode distances: {distances}")
            return distances

        elif attribute == "blood_type":
            # Match blood type format using regex
            # Return 1 if mismatch, 0 if exact match
            def extract_blood(s):
                match = re.search(r"(A\+|A-|B\+|B-|AB\+|AB-|O\+|O-)", s.upper())
                return match.group(0) if match else None

            return int(extract_blood(predicts) != extract_blood(true_answer))

        else:
            raise ValueError(f"Unknown attribute: {attribute}")

       
    def evalFPI(self, eval_args):
        
        keys = ["year_of_birth", "address_postcode", "social_insurance_number", "blood_type"]
        errors = {key: 0 for key in keys} # Record the attribute-wise scores.        
        count = {key: 0 for key in keys}
        results = [] # Collect model output
        for i in range(0, len(self.qa_data), self.eval_batch):
            
            batch = self.qa_data[i:i+self.eval_batch]
            attributes = [item["attribute"] for item in batch]
            questions = [self.Question_startToken + item["question"]+self.Question_endToken for item in batch]
            true_answers = [item["answer"] for item in batch]
            model_outputs = self.generate_answer(questions)
            print(model_outputs)
            
            for q, mo, ta, attr in zip(questions, model_outputs, true_answers, attributes):
                err = self.metric_FPI(mo, ta, attr)
                errors[attr] +=  err
                count[attr] += 1
                results.append({"attribute": attr, "question": q, "true_answer": ta, "model_output": mo, "error": err})
        
        for key in keys:
            errors[key] = errors[key]/count[key] if count[key] else errors[key]
        results.append(errors)
        results.append(count)
        
        if eval_args.modelType == 'unlearned':
            save_folder = f"{eval_args.unlearnSet}-lr{eval_args.lr_fgt}_WD{eval_args.wd_fgt}_loraRank{eval_args.LoRA_rank_fgt}_loraDrop{eval_args.lora_dropout_fgt}_reg{eval_args.reg_weights_fgt}/{eval_args.unlearn_method}"
            if not os.path.exists(os.path.join(eval_args.logDIR, save_folder)):
                os.makedirs(os.path.join(eval_args.logDIR, save_folder))
            save_fname =  f"epoch-{eval_args.eps_fgt}-{eval_args.datasetType}.json"
            save_fname = os.path.join(save_folder, save_fname)
        elif eval_args.modelType == 'learned':
            save_fname = f"lr{eval_args.lr}_WD{eval_args.weight_decay}_loraRank{eval_args.LoRA_rank}_loraDrop{eval_args.lora_dropout}/epoch-{eval_args.epochs}-{eval_args.datasetType}.json"
        else:
            save_fname = f"{eval_args.modelType}-{eval_args.datasetType}.json"


        with open(os.path.join(eval_args.logDIR, save_fname), "w") as f:
            json.dump(results, f, indent=4)


##### The avaible datasets ####
##### Please adjust by the real case ####
FILE_NAMES = {"train": "training_dataset.json", 
                "val": "validation_dataset.json",

             "forget": "forget.json", 
             "retain": "retain.json",
          "retain_sf": "retain-same_fn.json",
          "retain_sa": "retain-same_attr.json",
         "retain_sfa": "retain-same_fn_attr.json",
          "remain_sf": "remain-same_fn.json",
          "remain_sa": "remain-same_attr.json",
         "remain_sfa": "remain-same_fn_attr.json",

          "forget_df": "forget-diff_fn.json", 
          "retain_df": "retain-diff_fn.json",
       "retain_df_sf": "retain-diff_fn-same_fn.json",
       "retain_df_sa": "retain-diff_fn-same_attr.json",
      "retain_df_sfa": "retain-diff_fn-same_fn_attr.json",
       "remain_df_sf": "remain-diff_fn-same_fn.json",
       "remain_df_sa": "remain-diff_fn-same_attr.json",
      "remain_df_sfa": "remain-diff_fn-same_fn_attr.json",
      
          "forget_ri": "forget-rand_inst.json", 
          "retain_ri": "retain-rand_inst.json",
       "retain_ri_sf": "retain-rand_inst-same_fn.json",
       "retain_ri_sa": "retain-rand_inst-same_attr.json",
      "retain_ri_sfa": "retain-rand_inst-same_fn_attr.json",
       "remain_ri_sf": "remain-rand_inst-same_fn.json",
       "remain_ri_sa": "remain-rand_inst-same_attr.json",
      "remain_ri_sfa": "remain-rand_inst-same_fn_attr.json"}

def extract_dir(eval_args):
    file_path = "./data_generator/data"
    set_path = eval_args.unlearnSet
    filename = FILE_NAMES[eval_args.datasetType]
    dataDIR = os.path.join(file_path, set_path, filename)

    # Folders where finetuned model is saved. You can replace this by your own directory.    
    parent_folder = "fine_tuned_llama_7b"
    savefolder = f"lr{eval_args.lr}_WD{eval_args.weight_decay}_loraRank{eval_args.LoRA_rank}_loraDrop{eval_args.lora_dropout}/epoch-{eval_args.epochs}"
    learned_model_DIR = os.path.join(parent_folder, savefolder)
    modelDIR = {"learned": learned_model_DIR, "unlearned": None}

    if eval_args.modelType == 'unlearned':
        parent_folder = "unlearn_llama_7b"
        child_folder = f"{eval_args.unlearnSet}-lr{eval_args.lr_fgt}_WD{eval_args.wd_fgt}_loraRank{eval_args.LoRA_rank_fgt}_loraDrop{eval_args.lora_dropout_fgt}_reg{eval_args.reg_weights_fgt}"
        savefolder = f"{eval_args.unlearn_method}/epoch-{eval_args.eps_fgt}"
        unlearned_model_DIR = os.path.join(parent_folder, child_folder, savefolder)
        modelDIR["unlearned"] = unlearned_model_DIR

    return modelDIR, dataDIR

def main():
    
    parse = parser_eval()
    eval_args = parse.parse_args()
    modelDIR, dataDIR = extract_dir(eval_args)

    if eval_args.modelType == 'unlearned':
        eval_args.logDIR = "unlearn_llama_7b_log"
    # create folder to save evaluation result
    if not os.path.exists(eval_args.logDIR):
        os.makedirs(eval_args.logDIR)
    
    from saved_hf_key import HF_key  # Replace 'HF_key' by your own hugging face key.
    os.environ["HF_TOKEN"] = HF_key

    ####
    evaluator = EvalQA( 
        modelDIR = modelDIR,
        eval_batch = 5,  
        eval_args = eval_args,
        dataDIR = dataDIR, 
        model_name = eval_args.model_name, 
        auth_token = HF_key, 
        ) 
    
    evaluator.evalFPI(eval_args)


if __name__ == "__main__":
    main()