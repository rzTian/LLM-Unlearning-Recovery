import json
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
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

        quant_config = None
        if getattr(eval_args, "quant", "none") == "int8":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
        elif eval_args.quant == "int4":
            quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

        if eval_args.modelType == 'base': # For Llama model, set torch_dtype=torch.bfloat16 to avoid having NaN
            # self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.bfloat16, device_map="auto")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16 if quant_config is None else None,
                quantization_config=quant_config,
                device_map="auto"
            )
            print(f"[checkpoint]Load learned model from {self.model_name}")
        elif eval_args.modelType == 'learned':
            # self.base_model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.bfloat16, device_map="auto")
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16 if quant_config is None else None,
                quantization_config=quant_config,
                device_map="auto"
            )
            # Load fine-tuned LoRA adapters
            self.model = PeftModel.from_pretrained(self.base_model, self.modelDIR_learned)
            print(f"[checkpoint]Load learned model from {self.modelDIR_learned}")
        elif eval_args.modelType == 'unlearned':
            # self.base_model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.bfloat16, device_map="auto")
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16 if quant_config is None else None,
                quantization_config=quant_config,
                device_map="auto"
            )
            # Load fine-tuned LoRA adapters
            self.ref_model = PeftModel.from_pretrained(self.base_model, self.modelDIR_learned)
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
        """
        Calculate error metrics for FPI (Personally Identifiable Information) attributes 
        based on specific formulas for each attribute type.
        
        Args:
            predicts (str): Predicted value from the model
            true_answer (str): Ground truth value
            attribute (str): Type of attribute to evaluate (e.g., "year_of_birth", "social_insurance_number")
        
        Returns:
            float: Normalized error value according to the attribute-specific formula
        """
        predicts = predicts.strip()
        true_answer = true_answer.strip()

        # Helper function to calculate Hamming distance (for postcode comparison)
        def hamming_distance(s1, s2):
            """
            Calculate Hamming distance between two strings.
            
            Args:
                s1 (str): First string to compare
                s2 (str): Second string to compare
                
            Returns:
                int: Number of positions where characters differ
            """
            if len(s1) != len(s2):
                return 6
            return sum(c1 != c2 for c1, c2 in zip(s1, s2))

        if attribute == "year_of_birth":
            # Extract 4-digit year from string (returns 0 if no digits found)
            def extract_year(s):
                year_match = re.search(r"\b(19|20)\d{2}\b", s)
                if year_match:
                    return int(year_match.group())
                return 0  # 默认返回0
                
            # Error formula: min(|predicted_year - true_year|, 20) / 20
            absolute_error = abs(extract_year(predicts) - extract_year(true_answer))
            return min(absolute_error, 20) / 20

        elif attribute == "social_insurance_number":
            # Extract first 9 digits from string
            def extract_digits(s): 
                return ''.join(re.findall(r"\d", s))[:9]
                
            # Error formula: Levenshtein edit distance / 9
            edit_distance = Levenshtein.distance(extract_digits(predicts), extract_digits(true_answer))
            return edit_distance / 9

        elif attribute == "address_postcode":
            # Extract 6-character alphanumeric postcode (case-insensitive)
            def extract_postcodes(s): return re.findall(r"(?:[A-Z][0-9]){3}", s)
            candidates = extract_postcodes(predicts)
            if not candidates:
                return 1.0
            true_code = extract_postcodes(true_answer)[0]
                
            # Error formula: Hamming distance / 6
            hamming_dist = hamming_distance(candidates[0], true_code)
            return hamming_dist / 6

        elif attribute == "blood_type":
            # Extract standard blood type format (e.g., A+, B-, AB+, O-)
            def extract_blood(s): 
                match = re.search(r"(A\+|A-|B\+|B-|AB\+|AB-|O\+|O-)", s.upper())
                return match.group(0) if match else None
                
            # Error formula: Indicator function I(y ≠ y*) (1 if mismatch, 0 if match)
            return int(extract_blood(predicts) != extract_blood(true_answer))

        else:
            raise ValueError(f"Unknown attribute type: {attribute}")

       
    def evalFPI(self, eval_args):
        # Evaluate common knowledge
        if eval_args.datasetType == 'common':
            return self.eval_common(eval_args)
        
        # Evaluate forgetting performance
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
            save_folder = f"{eval_args.unlearnSet}-lr{eval_args.lr_fgt}_WD{eval_args.wd_fgt}_loraRank{eval_args.LoRA_rank_fgt}_loraDrop{eval_args.lora_dropout_fgt}_GradStep{eval_args.grad_acc_steps_fgt}_reg{eval_args.reg_weights_fgt}/{eval_args.unlearn_method}"
            if not os.path.exists(os.path.join(eval_args.logDIR, save_folder)):
                os.makedirs(os.path.join(eval_args.logDIR, save_folder))
            save_fname = f"epoch-{eval_args.eps_fgt}-{eval_args.datasetType}.json"
            save_fname = os.path.join(save_folder, save_fname)
        elif eval_args.modelType == 'learned':
            save_folder = f"lr{eval_args.lr}_WD{eval_args.weight_decay}_loraRank{eval_args.LoRA_rank}_loraDrop{eval_args.lora_dropout}_GradStsp{eval_args.grad_acc_steps}/{eval_args.unlearnSet}"
            if not os.path.exists(os.path.join(eval_args.logDIR, save_folder)):
                os.makedirs(os.path.join(eval_args.logDIR, save_folder))
            save_fname = f"epoch-{eval_args.epochs}-{eval_args.datasetType}.json"
            save_fname = os.path.join(save_folder, save_fname)
        else:
            save_fname = f"{eval_args.modelType}-{eval_args.datasetType}.json"


        with open(os.path.join(eval_args.logDIR, save_fname), "w") as f:
            json.dump(results, f, indent=4)
    
    # evaluate common knowledge
    def _clean_model_output(self, prompts, outputs):
        cleaned_outputs = []
        EMPTY_MARK = "[EMPTY]"
        start_token = self.Question_startToken
        end_token = self.Question_endToken
        
        for prompt, output in zip(prompts, outputs):
            # 步骤1：先移除【完整的Prompt标记对】（优先处理完整结构）
            complete_token_pattern = re.compile(
                re.escape(start_token) + r".*?" + re.escape(end_token),
                re.DOTALL
            )
            while complete_token_pattern.search(output):
                output = complete_token_pattern.sub("", output)

            # 步骤2：处理【孤立startToken及其后续内容】（核心新增逻辑）
            # 逻辑：若存在未匹配的startToken（start数>end数），删除最后一个startToken及其后面的所有内容
            # 统计当前output中startToken和endToken的数量（判断是否有未匹配的startToken）
            start_count = output.count(start_token)
            end_count = output.count(end_token)
            
            if start_count > end_count:  # 存在未匹配的startToken
                # 找到最后一个startToken的位置
                last_start_idx = output.rfind(start_token)
                if last_start_idx != -1:  # 确保找到startToken
                    # 删除最后一个startToken及其后面的所有内容
                    output = output[:last_start_idx].strip()  # 保留startToken之前的内容

            # 步骤3：清理【孤立的标记碎片】（剩余的单个start/endToken）
            # 仅匹配孤立的startToken或endToken（无对应另一半）
            incomplete_token_pattern = re.compile(
                r"(?<!{}){}|{}(?!{})".format(
                    re.escape(end_token), re.escape(start_token),
                    re.escape(end_token), re.escape(start_token)
                ),
                re.IGNORECASE
            )
            output = incomplete_token_pattern.sub("", output)

            # 步骤4：移除【与Prompt重复的片段】+【连续重复文本】
            # 清理Prompt中的标记，用于匹配重复
            clean_prompt = complete_token_pattern.sub("", prompt).strip()
            if clean_prompt and len(clean_prompt) > 5:
                prompt_pattern = re.compile(r"\s*" + re.escape(clean_prompt) + r"\s*", re.IGNORECASE)
                output = prompt_pattern.sub("", output)
            
            # 移除连续重复文本
            repeat_pattern = re.compile(r"(\b.+\b)(\s*\1){2,}", re.DOTALL)
            output = repeat_pattern.sub(r"\1", output)

            # 步骤5：标准化【空内容】
            cleaned_lines = [line.strip() for line in output.splitlines() if line.strip()]
            output = "\n".join(cleaned_lines)
            cleaned = output if output else EMPTY_MARK

            cleaned_outputs.append(cleaned)

        return cleaned_outputs

    def eval_common(self, eval_args):
        """Batch process common knowledge evaluation (imitating evalFPI batch logic)"""
        baseLogJSON = os.path.join(eval_args.logDIR, "base.json") # base-common.json
        results = []

        # Initialize LLM judge (reuse singleton to avoid repeated loading)
        if not hasattr(self, 'llm_judge'):
            from llm_judge import LLMJudge
            self.llm_judge = LLMJudge()
            print(f"[INFO] LLM Judge initialized (model: {self.llm_judge.model_name})")
        
        # Store base answer mapping (only used for non-base models)
        base_info_map = None
        if eval_args.modelType != 'base':
            base_answers = self.load_dataset(dataDIR=baseLogJSON)
            if not base_answers:
                print("[ERROR] Base model results not found! Run base evaluation first.")
                return
            # Filter statistical items: only keep valid samples with "name" field (exclude statistical block at the end)
            valid_base_answers = [item for item in base_answers if "name" in item]
            # Build mapping table (only once)
            base_info_map = {
                (item["name"], item["attribute"], item["question"]): {
                    "answer": item["base_answer"],
                    "score": item["base_llm_score"]
                } 
                for item in valid_base_answers
            }
            print(f"[INFO] Loaded {len(valid_base_answers)} valid base answers (filtered stats item)")

        
        # Calculate total batches for progress tracking
        total_samples = len(self.qa_data)
        total_batches = (total_samples + self.eval_batch - 1) // self.eval_batch  # Ceiling division
        print(f"[INFO] Starting evaluation: {total_samples} samples, {total_batches} batches (batch size: {self.eval_batch})")
  
        if eval_args.modelType == 'base':
            # Handle all base model logic
            self.eval_common_base_model(eval_args, total_samples, total_batches)
            return  # Return directly after base model processing is complete

        for batch_idx in range(0, total_samples, self.eval_batch):
            # Calculate current batch range
            batch_start = batch_idx
            batch_end = min(batch_idx + self.eval_batch, total_samples)
            current_batch_num = (batch_start // self.eval_batch) + 1  # 1-based batch number

            # Load & process batch data
            batch = self.qa_data[batch_start:batch_end]
            raw_questions = [item["question"] for item in batch]
            print(f"\n[PROGRESS] Batch {current_batch_num}/{total_batches}: Processing samples {batch_start+1}-{batch_end}")

            # Step 1: Generate model outputs
            formatted_questions = [
                f"{self.Question_startToken}{q}{self.Question_endToken}" 
                for q in raw_questions
            ]
            model_outputs = self.generate_answer(formatted_questions)
            cleaned_outputs = self._clean_model_output(raw_questions, model_outputs)
            
            # Step 3: Process NON-BASE model (compare with base)
            # Match current batch with base answers
            batch_data = []
            for item, output, q in zip(batch, cleaned_outputs, raw_questions):
                key = (item["name"], item["attribute"], q)
                if key in base_info_map:
                    batch_data.append((item, output, q, base_info_map[key]))
                
            # Skip if no matching base answers
            if not batch_data:
                print(f"[WARN] Batch {current_batch_num} (NON-BASE): No matching base answers → Skipping")
                continue
                
            # Unpack batch data
            items, gens, qs, base_datas = zip(*batch_data)
            gt_answers = [d["answer"] for d in base_datas]
            gt_scores = [d["score"] for d in base_datas]

            # Get metric flags
            calc_model_base = getattr(eval_args, "calc_model_base", False)
            calc_full_llm = getattr(eval_args, "calc_full_llm", True)
            calc_text_metrics = getattr(eval_args, "calc_text_metrics", False)

            # Calculate metrics
            model_scores = [None]*len(batch_data)
            llm_relscr = [None]*len(batch_data)
            if calc_model_base:
                model_res = self.llm_judge.judge_base_batch(qs, gens)
                model_scores = [r["score"] for r in model_res]
                llm_relscr = [round(m - g, 2) for m, g in zip(model_scores, gt_scores)]

            full_scores = [None]*len(batch_data)
            full_evals = [None]*len(batch_data)
            fpi_attributes = ["year_of_birth", "address_postcode", "social_insurance_number", "blood_type"]
            if calc_full_llm:
                full_res = None
                # 遍历每个样本，根据属性类型选择评估方式
                for i in range(len(batch_data)):
                    item = items[i]
                    gen = gens[i]
                    gt = gt_answers[i]
                    attr = item["attribute"]
                    # 如果是FPI相关属性，使用metric_FPI计算分数
                    if attr in fpi_attributes:
                        # 调用metric_FPI计算误差（注意：这里将误差转换为分数，1-误差值）
                        error = self.metric_FPI(gen, gt, attr)
                        full_scores[i] = round(10 - error * 9, 1)  # 转换为分数（0-10范围）
                        full_evals[i] = f"FPI metric used for {attr}: error={error:.4f}"
                    else: 
                        if full_res is None:  # 延迟计算LLM结果，避免不必要的调用
                            full_res = self.llm_judge.judge_batch(qs, gt_answers, gens)
                        full_scores[i] = full_res[i]["score"]
                        full_evals[i] = full_res[i]["evaluation_text"]

            metrics_list = [None]*len(batch_data)
            if calc_text_metrics:
                metrics_list = self.eval_gen_text_metrics(gens, gt_answers, qs)

            # Assemble & save results
            for idx in range(len(batch_data)):
                res = {
                    "name": items[idx]["name"],
                    "attribute": items[idx]["attribute"],
                    "question": qs[idx],
                    "base_answer": gt_answers[idx],
                    "model_output": gens[idx],
                    "base_llm_score": gt_scores[idx]
                }
                if calc_model_base:
                    res.update({"model_llm_score": model_scores[idx], "llm_relscr": llm_relscr[idx]})
                if calc_full_llm:
                    res.update({"llm_score": full_scores[idx], "llm_eval": full_evals[idx]})
                if calc_text_metrics:
                    res["metrics"] = metrics_list[idx]
                results.append(res)
        
        print(f"\n[FINAL SUMMARY] Evaluation completed! Total samples processed: {len(results)}")
        
        # 统计均分
        def avg(scores):
            """计算有效分数的平均值（排除None和-1.0）"""
            valid = [s for s in scores if s is not None and s != -1.0]
            return round(sum(valid)/len(valid), 2) if valid else None
        
        if not results:
            return results

        # 收集指标
        metrics = {"base_llm_score": [r["base_llm_score"] for r in results]}
        metrics.update({
            "model_llm_score": [r.get("model_llm_score") for r in results],
            "llm_relscr": [r.get("llm_relscr") for r in results],
            "llm_score": [r.get("llm_score") for r in results]
        })
        # 收集metrics子指标
        sub_metrics = {}
        for r in results:
            if "metrics" in r and isinstance(r["metrics"], dict):
                for k, v in r["metrics"].items():
                    if isinstance(v, (int, float)):
                        sub_metrics.setdefault(k, []).append(v)

        # 生成统计块
        stats = {
            "type": "statistics",
            "model_type": eval_args.modelType,
            "total_samples": len(results),
            "average_scores": {k: avg(v) for k, v in metrics.items() if avg(v) is not None}
        }
        if sub_metrics:
            stats["average_scores"]["metrics"] = {k: avg(v) for k, v in sub_metrics.items() if avg(v) is not None}

        results.append(stats)
                
        # Save results
        if eval_args.modelType == 'unlearned':
            save_folder = (
                f"{eval_args.unlearnSet}-lr{eval_args.lr_fgt}_WD{eval_args.wd_fgt}_"
                f"loraRank{eval_args.LoRA_rank_fgt}_loraDrop{eval_args.lora_dropout_fgt}_"
                f"GradStep{eval_args.grad_acc_steps_fgt}_"
                f"reg{eval_args.reg_weights_fgt}/{eval_args.unlearn_method}"
            )
            save_fname = f"epoch-{eval_args.eps_fgt}-{eval_args.datasetType}.json"
        elif eval_args.modelType == 'learned':
            save_folder = (
                f"lr{eval_args.lr}_WD{eval_args.weight_decay}_loraRank{eval_args.LoRA_rank}_"
                f"loraDrop{eval_args.lora_dropout}_GradStsp{eval_args.grad_acc_steps}"
            )
            save_fname = f"epoch-{eval_args.epochs}-{eval_args.datasetType}.json"
            
        save_path = os.path.join(eval_args.logDIR, save_folder, save_fname)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Common evaluation results saved to: {save_path}")

    def eval_common_base_model(self, eval_args, total_samples, total_batches):
        """Handle all evaluation logic for base model"""
        import random
        results = []
        
        # Generate irrelevant prompts to let base model create unrelated answers
        from llm_prompt import IRRELEVANT_PROMPTS
        print(f"[INFO] Using {len(IRRELEVANT_PROMPTS)} pre-defined irrelevant prompts for base model")
        
        for batch_idx in range(0, total_samples, self.eval_batch):
            # Calculate current batch range
            batch_start = batch_idx
            batch_end = min(batch_idx + self.eval_batch, total_samples)
            current_batch_num = (batch_start // self.eval_batch) + 1  # 1-based batch number

            # Load & process batch data
            batch = self.qa_data[batch_start:batch_end]
            raw_questions = [item["question"] for item in batch]
            print(f"\n[PROGRESS] Batch {current_batch_num}/{total_batches}: Processing samples {batch_start+1}-{batch_end}")

            # Step 1: Generate model outputs (normal answers)
            formatted_questions = [
                f"{self.Question_startToken}{q}{self.Question_endToken}" 
                for q in raw_questions
            ]
            model_outputs = self.generate_answer(formatted_questions)
            cleaned_outputs = self._clean_model_output(raw_questions, model_outputs)
            
            # Step 2: Generate answers irrelevant to the questions (using base model)
            # Randomly select irrelevant prompts
            random_prompts = [random.choice(IRRELEVANT_PROMPTS) for _ in range(len(raw_questions))]
            formatted_random_prompts = [
                f"{self.Question_startToken}{p}{self.Question_endToken}" 
                for p in random_prompts
            ]
            # Generate irrelevant answers using base model
            irrelevant_outputs = self.generate_answer(formatted_random_prompts)
            cleaned_irrelevant_outputs = self._clean_model_output(random_prompts, irrelevant_outputs)
            
            # ------------------------------
            # Task 1: Basic Evaluation (save to base.json)
            # ------------------------------
            llm_results = self.llm_judge.judge_base_batch(
                questions=raw_questions,
                answers=cleaned_outputs
            )
            # Collect basic evaluation results
            base_results = []
            for item, output, llm_res in zip(batch, cleaned_outputs, llm_results):
                base_results.append({
                    "name": item["name"],
                    "attribute": item["attribute"],
                    "question": item["question"],
                    "base_answer": output,
                    "base_llm_score": llm_res["score"],
                    "base_llm_eval": llm_res["evaluation_text"]
                })
            
            # 2. Self-comparison evaluation - save to base-10.json
            full_res_10 = self.llm_judge.judge_batch(
                questions=raw_questions,
                references=cleaned_outputs,
                model_answers=cleaned_outputs
            )
            base_10_results = []
            for item, ref, ans, res in zip(batch, cleaned_outputs, cleaned_outputs, full_res_10):
                base_10_results.append({
                    "name": item["name"],
                    "attribute": item["attribute"],
                    "question": item["question"],
                    "reference": ref,
                    "model_answer": ans,
                    "score": res["score"],
                    "evaluation_text": res["evaluation_text"]
                })
            
            # 3. Garbled text comparison evaluation - save to base-0.json
            # Generate garbled text (using random Unicode characters)
            garbled_outputs = [''.join([chr(random.randint(0x00, 0xFF)) for _ in range(len(output))]) 
                            for output in cleaned_outputs]
            full_res_0 = self.llm_judge.judge_batch(
                questions=raw_questions,
                references=cleaned_outputs,
                model_answers=garbled_outputs
            )
            base_0_results = []
            for item, ref, ans, res in zip(batch, cleaned_outputs, garbled_outputs, full_res_0):
                base_0_results.append({
                    "name": item["name"],
                    "attribute": item["attribute"],
                    "question": item["question"],
                    "reference": ref,
                    "model_answer": ans,
                    "score": res["score"],
                    "evaluation_text": res["evaluation_text"]
                })
            
            # 4. Irrelevant answer comparison evaluation - save to base-1.json
            full_res_1 = self.llm_judge.judge_batch(
                questions=raw_questions,
                references=cleaned_outputs,
                model_answers=cleaned_irrelevant_outputs
            )
            base_1_results = []
            for item, ref, ans, res in zip(batch, cleaned_outputs, cleaned_irrelevant_outputs, full_res_1):
                base_1_results.append({
                    "name": item["name"],
                    "attribute": item["attribute"],
                    "question": item["question"],
                    "reference": ref,
                    "model_answer": ans,
                    "score": res["score"],
                    "evaluation_text": res["evaluation_text"]
                })
            
            # Combine all results for current batch
            results.append({
                "batch": current_batch_num,
                "base_results": base_results,
                "base_10_results": base_10_results,
                "base_0_results": base_0_results,
                "base_1_results": base_1_results
            })
        
        print(f"\n[FINAL SUMMARY] Base model evaluation completed!")
        
        # Calculate average score
        def avg(scores):
            """Calculate average of valid scores (excluding None and -1.0)"""
            valid = [s for s in scores if s is not None and s != -1.0]
            return round(sum(valid)/len(valid), 2) if valid else None
        
        if not results:
            return
        
        # Combine results from all batches
        all_base = []
        all_base_10 = []
        all_base_0 = []
        all_base_1 = []
        
        for batch in results:
            all_base.extend(batch["base_results"])
            all_base_10.extend(batch["base_10_results"])
            all_base_0.extend(batch["base_0_results"])
            all_base_1.extend(batch["base_1_results"])
        
        # Add statistics to each result set
        def add_statistics(data, model_type_suffix):
            metrics = {"score": [r["score"] if "score" in r else r["base_llm_score"] for r in data]}
            stats = {
                "type": "statistics",
                "model_type": f"base-{model_type_suffix}",
                "total_samples": len(data),
                "average_scores": {k: avg(v) for k, v in metrics.items() if avg(v) is not None}
            }
            data_with_stats = data.copy()
            data_with_stats.append(stats)
            return data_with_stats
        
        # Save to different files
        save_paths = {
            "base": os.path.join(eval_args.logDIR, "base.json"),
            "base-10": os.path.join(eval_args.logDIR, "base-10.json"),
            "base-0": os.path.join(eval_args.logDIR, "base-0.json"),
            "base-1": os.path.join(eval_args.logDIR, "base-1.json")
        }
        
        # Process and save each file
        all_base_with_stats = add_statistics(all_base, "")
        all_base_10_with_stats = add_statistics(all_base_10, "10")
        all_base_0_with_stats = add_statistics(all_base_0, "0")
        all_base_1_with_stats = add_statistics(all_base_1, "1")
        
        # Save files
        with open(save_paths["base"], "w") as f:
            json.dump(all_base_with_stats, f, indent=4)
        print(f"Base model basic evaluation results saved to: {save_paths['base']}")
        
        with open(save_paths["base-10"], "w") as f:
            json.dump(all_base_10_with_stats, f, indent=4)
        print(f"Base model self-comparison results saved to: {save_paths['base-10']}")
        
        with open(save_paths["base-0"], "w") as f:
            json.dump(all_base_0_with_stats, f, indent=4)
        print(f"Base model garbled text comparison results saved to: {save_paths['base-0']}")
        
        with open(save_paths["base-1"], "w") as f:
            json.dump(all_base_1_with_stats, f, indent=4)
        print(f"Base model irrelevant answer comparison results saved to: {save_paths['base-1']}")

    def eval_gen_text_metrics(self, gen_outputs, ground_truths, questions):
        """
        同时接收生成文本、参考文本和问题列表，计算评估指标
        输入：三个长度相同的列表
        输出：与输入长度相同的指标列表，一一对应
        """
        import nltk
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from rouge_score import rouge_scorer

        # 确保输入为列表
        if isinstance(gen_outputs, str):
            gen_outputs = [gen_outputs]
        if isinstance(ground_truths, str):
            ground_truths = [ground_truths]
        if isinstance(questions, str):
            questions = [questions]
        
        if not (len(gen_outputs) == len(ground_truths) == len(questions)):
            raise ValueError(f"生成文本({len(gen_outputs)})、参考文本({len(ground_truths)})、问题({len(questions)})长度必须一致")

        # 初始化评估工具
        smoothing = SmoothingFunction().method4
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        
        # 存储每个样本的指标（与输入列表长度一一对应）
        all_metrics = []
        
        for gen, gt, q in zip(gen_outputs, ground_truths, questions):
            # 1. BLEU计算
            gen_tokens = nltk.word_tokenize(gen.lower())
            gt_tokens = [nltk.word_tokenize(gt.lower())]
            bleu_score = sentence_bleu(
                references=gt_tokens,
                hypothesis=gen_tokens,
                smoothing_function=smoothing,
                weights=(0.25, 0.25, 0.25, 0.25)
            )
            
            # 2. ROUGE计算
            scores = scorer.score(gt, gen)
            rouge1 = {
                'precision': round(scores['rouge1'].precision, 4),
                'recall': round(scores['rouge1'].recall, 4),
                'fmeasure': round(scores['rouge1'].fmeasure, 4)
            }
            rougeL = {
                'precision': round(scores['rougeL'].precision, 4),
                'recall': round(scores['rougeL'].recall, 4),
                'fmeasure': round(scores['rougeL'].fmeasure, 4)
            }
            
            # 3. 新增：问题相关性（生成文本与问题的ROUGE）
            q_scores = scorer.score(q, gen)
            q_relevance = round(q_scores['rougeL'].fmeasure, 4)
            
            # 单个样本的所有指标
            sample_metrics = {
                'bleu': round(bleu_score, 4),
                'rouge1': rouge1,
                'rougeL': rougeL,
                'question_relevance': q_relevance
            }
            all_metrics.append(sample_metrics)
        
        return all_metrics

##### The avaible datasets ####
##### Please adjust by the real case ####
FILE_NAMES = {"train": "training_dataset.json", 
                "val": "validation_dataset.json",
            "train_t": "training_testset.json", 
              "val_t": "validation_testset.json",
             "common": "common_knowledge_questions.json",

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
    if eval_args.datasetType in ["train", "val", "common"]:
        set_path = ""
        filename = FILE_NAMES[eval_args.datasetType]
    else:
        set_path = eval_args.unlearnSet
        filename = "test-" + FILE_NAMES[eval_args.datasetType]
    dataDIR = os.path.join(file_path, set_path, filename)

    if eval_args.datasetType not in FILE_NAMES:
        raise ValueError(f"Unknown datasetType: {eval_args.datasetType}")

    # Folders where finetuned model is saved. You can replace this by your own directory.    
    parent_folder = "fine_tuned_deepseek_7b"
    savefolder = f"lr{eval_args.lr}_WD{eval_args.weight_decay}_loraRank{eval_args.LoRA_rank}_loraDrop{eval_args.lora_dropout}_GradStsp{eval_args.grad_acc_steps}/epoch-{eval_args.epochs}"
    learned_model_DIR = os.path.join(parent_folder, savefolder)
    modelDIR = {"learned": learned_model_DIR, "unlearned": None}

    if eval_args.modelType == 'unlearned':
        parent_folder = "unlearn_deepseek_7b"
        child_folder = f"{eval_args.unlearnSet}-lr{eval_args.lr_fgt}_WD{eval_args.wd_fgt}_loraRank{eval_args.LoRA_rank_fgt}_loraDrop{eval_args.lora_dropout_fgt}_GradStep{eval_args.grad_acc_steps_fgt}_reg{eval_args.reg_weights_fgt}"
        if eval_args.beta_fgt != 0.1:
            child_folder += f"_beta{eval_args.beta_fgt}"
        savefolder = f"{eval_args.unlearn_method}/epoch-{eval_args.eps_fgt}"
        unlearned_model_DIR = os.path.join(parent_folder, child_folder, savefolder)
        modelDIR["unlearned"] = unlearned_model_DIR

    return modelDIR, dataDIR

def main():
    
    parse = parser_eval()
    eval_args = parse.parse_args()
    modelDIR, dataDIR = extract_dir(eval_args)

    if eval_args.modelType == 'learned':
        eval_args.logDIR = "fine_tuned_deepseek_7b_log"
    elif eval_args.modelType == 'unlearned':
        eval_args.logDIR = "unlearn_deepseek_7b_log"
    
    if eval_args.datasetType == 'common':
        eval_args.logDIR = "base_deepseek_7b_log"

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