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




class recoverQA(EvalQA):
    
    def __init__(self, flip_logit, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.digit_pred_processor = CustomizedLogitsProcessor(
                                                        tokenizer = self.tokenizer, 
                                                        tokenType = "digits", 
                                                        flip_logit=flip_logit)
        self.bloodType_pred_processor = CustomizedLogitsProcessor(
                                                        tokenizer = self.tokenizer, 
                                                        tokenType = "blood_type", 
                                                        flip_logit=flip_logit)


    def generate_answer(self, questions, output_length, predict_bloodType=True):


        inputs = self.tokenizer(questions, padding=True, truncation=True, max_length=150, return_tensors="pt").to(self.device)
        logits_processor = self.digit_pred_processor if not predict_bloodType else self.bloodType_pred_processor

        with torch.no_grad():
            output = self.model.generate(
                                **inputs, 
                                generation_config=self.gen_cfg, 
                                tokenizer=self.tokenizer,
                                logits_processor = [logits_processor],
                                output_logits=True,   ####
                                return_dict_in_generate=True, ###
                                )

        print("selected_token_ids:", logits_processor.selected_token_ids)
        print("logits:", torch.cat(output.logits, dim=0)[:, logits_processor.selected_token_ids])
        decoded_outputs = self.tokenizer.batch_decode(output.sequences, skip_special_tokens=False) #####       
        print("token_ids:", output.sequences)
        print("decoded:", decoded_outputs)

        # decoded_outputs = self.tokenizer.batch_decode(output, skip_special_tokens=False)
            
        if not predict_bloodType:
            # Decode and extract first N digits from each output
            truncated_outputs = [''.join(re.findall(r"\d", text))[:l] for text, l in zip(decoded_outputs, output_length)]
            return truncated_outputs
        else:
            return decoded_outputs
    
    def mask_info(self, text):
        blood_types = ["A\\+\\.?","A-\\.?","B\\+\\.?","B-\\.?","AB\\+\\.?","AB-\\.?","O\\+\\.?","O-\\.?"]
        pattern_blood = "|".join(blood_types)
        # Remove blood types
        text = re.sub(pattern_blood, "", text)
        # Remove digits and periods
        text = re.sub(r"[0-9\.]", "", text)

        # Clean up extra spaces
        # text = re.sub(r'\s+', ' ', text).strip()
        return text



    def evalFPI(self, eval_args):
        
        keys = ["year_of_birth", "credit_card_number", "credit_card_cvv", "annual_income", "blood_type"]
        # Control the output length for digits predictions according to the attributes
        lengths = [4,16,3,5]
        attr_lens = {key: value for key, value in zip(keys[:4], lengths)}

        # Record the attribute-wise scores.
        errors = {key: 0 for key in keys}         
        count = {key: 0 for key in keys}
        results = [] # Collect model output
        
        for i in range(0, len(self.qa_data), self.eval_batch):
            
            batch = self.qa_data[i:i+self.eval_batch]
            # Collect all the questions related to predicting digits            
            attributes_digits = []
            questions_digits = []
            true_answers_digits = []
            model_outputs_digits = []
            # Collect the questions related to predicting alphabets (i.e., bloodtype)
            attributes_alphabet = []
            questions_alphabet = []
            true_answers_alphabet = []
            model_outputs_alphabet = []

            for item in batch:
                question = self.Question_startToken + item["question"] + self.Question_endToken
                ### Modify the original question by adding a "leading" sentence.
                question = question + " " + self.mask_info(item["answer"]) # + self.tokenizer.bos_token + " "
                if item["attribute"] == "blood_type":                
                    questions_alphabet.append(question)
                    true_answers_alphabet.append(item["answer"])
                else:
                    questions_digits.append(question)
                    attributes_digits.append(item["attribute"])
                    true_answers_digits.append(item["answer"])
            
            output_length = [attr_lens[attr] for attr in attributes_digits] 
            if questions_digits:                
                model_outputs_digits = self.generate_answer(questions=questions_digits, output_length=output_length, predict_bloodType=False)
            if questions_alphabet:
                model_outputs_alphabet = self.generate_answer(questions=questions_alphabet, output_length=None, predict_bloodType=True)
                attributes_alphabet = ["blood_type"]*len(questions_alphabet)

            questions  = questions_digits+questions_alphabet
            model_outputs = model_outputs_digits + model_outputs_alphabet
            true_answers = true_answers_digits + true_answers_alphabet
            attributes = attributes_digits + attributes_alphabet

            for q, mo, ta, attr in zip(questions, model_outputs, true_answers, attributes):
                err = self.metric_FPI(mo, ta, attr)
                errors[attr] +=  err
                count[attr] += 1
                results.append({"attribute": attr, "question": q, "true_answer": ta, "model_output": mo, "error": err})

        
        for key in keys:
            errors[key] = errors[key]/count[key] if count[key] else errors[key]
        results.append(errors)
        results.append(count)
        
        ### Feel free to modify the file name for saving the result.
        if eval_args.modelType == 'unlearned':
            save_folder = f"{eval_args.modelType}-num_fgt{eval_args.num_fgt}-lr{eval_args.lr_fgt}-WD{eval_args.wd_fgt}_loraRank{eval_args.LoRA_rank_fgt}_loraDrop{eval_args.lora_dropout_fgt}_eps{eval_args.eps_fgt}_reg{eval_args.reg_weights_fgt}"
            if not os.path.exists(os.path.join(eval_args.logDIR, save_folder)):
                os.makedirs(os.path.join(eval_args.logDIR, save_folder))
            save_fname =  f"{eval_args.unlearn_method}-{eval_args.datasetType}-recovery-flip_logit-{eval_args.flip_logit}.json"
            save_fname = os.path.join(save_folder, save_fname)
        else:
            save_fname =  f"{eval_args.modelType}-{eval_args.datasetType}-recovery-flip_logit-{eval_args.flip_logit}.json"

        with open(os.path.join(eval_args.logDIR, save_fname), "w") as f:
            json.dump(results, f, indent=4)



def main():

    from evaluate import FILE_NAMES

    parse = parser_eval()
    parse.add_argument('--flip_logit', default=1, type=int)
    eval_args = parse.parse_args()

    file_path = "./data_generator/data"
    filename = FILE_NAMES[eval_args.datasetType]  
    dataDIR = os.path.join(file_path, filename)
   
    from saved_hf_key import HF_key  # Replace 'HF_key' by your own hugging face key.
    os.environ["HF_TOKEN"] = HF_key

    parent_folder = "fine_tuned_llama_7b"
    savefolder = f"lr{eval_args.lr}_eps{eval_args.epochs}_WD{eval_args.weight_decay}_loraRank{eval_args.LoRA_rank}_loraDrop{eval_args.lora_dropout}"
    learned_model_DIR = os.path.join(parent_folder, savefolder)
    modelDIR = {"learned": learned_model_DIR, "unlearned": None}

    if eval_args.modelType == 'unlearned':
        parent_folder = "unlearn_llama_7b-1"
        child_folder = f"num_fgt{eval_args.num_fgt}-lr{eval_args.lr_fgt}_WD{eval_args.wd_fgt}_loraRank{eval_args.LoRA_rank_fgt}_loraDrop{eval_args.lora_dropout_fgt}_eps{eval_args.eps_fgt}_reg{eval_args.reg_weights_fgt}"
        savefolder = f"{eval_args.unlearn_method}"
        unlearned_model_DIR = os.path.join(parent_folder, child_folder, savefolder)
        modelDIR["unlearned"] = unlearned_model_DIR
        eval_args.logDIR = "recovery_llama_7b_log-1"
        
    
    # create logger folder
    if not os.path.exists(eval_args.logDIR):
        os.makedirs(eval_args.logDIR)

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