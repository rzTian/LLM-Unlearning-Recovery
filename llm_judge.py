import torch
import time
import re
import os
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
from llm_prompt import REF_EVAL_PROMPT, BASE_EVAL_PROMPT


class LLMJudge:
    def __init__(self,
                 model_name: str = "Qwen/Qwen3-8B",
                 use_quantization: bool = True,
                 max_new_tokens: int = 800):
        self.model_name = model_name
        self.use_quantization = use_quantization
        self.max_new_tokens = max_new_tokens
        self.default_score = -1.0

        self.prompt_templates = {
            "ref_eval": REF_EVAL_PROMPT,
            "base_eval": BASE_EVAL_PROMPT
        }

        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()

    def _load_tokenizer(self) -> AutoTokenizer:
        path = os.path.expanduser(self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(
            path,
            trust_remote_code=True,
            local_files_only=True,
            padding_side="right",
        )
        if tokenizer.pad_token is None:
            if "<|pad|>" in tokenizer.get_vocab():
                tokenizer.pad_token = "<|pad|>"
            else:
                tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        return tokenizer

    def _load_model(self) -> AutoModelForCausalLM:
        path = os.path.expanduser(self.model_name)
        cfg = AutoConfig.from_pretrained(
            path, trust_remote_code=True, local_files_only=True
        )

        model_kwargs = {
            "config": cfg,
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "trust_remote_code": True,
            "local_files_only": True
        }

        if self.use_quantization:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

        model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        model.eval()
        return model

    def _parse_score(self, raw_output: str) -> float:
        """Parse Rating: num / [num] / [[num]] formats."""
        cleaned_output = re.sub(r'[^\x00-\x7F]+', '', raw_output)
        cleaned_output = re.sub(r'\s+', ' ', cleaned_output).strip()

        score_pattern = r"(Final\s+)?Rating:\s*(\[\[|\[)?(\d+(\.\d+)?)(\]\]|\])?"
        match = re.search(score_pattern, cleaned_output, re.IGNORECASE)

        if match:
            score_str = match.group(3)
            try:
                score = round(float(score_str), 1)
                if 0.0 <= score <= 10.0:
                    return score
            except ValueError:
                return self.default_score
        
        return self.default_score

    def _release_memory(self, sleep_time: float = 0.2) -> None:
        torch.cuda.empty_cache()
        time.sleep(sleep_time)

    def _generate_response(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        return self.tokenizer.decode(
            outputs[0][len(input_ids[0]):],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
    
    def format_empty(self, text: str) -> str:
        empty_mark = "[EMPTY]"
        if not text or text.strip() == "":
            return empty_mark
        return text.strip()

    def judge_base_single(self, question: str, answer: str) -> Dict:
        sys_prompt = self.prompt_templates["base_eval"]["system"]
        usr_prompt = self.prompt_templates["base_eval"]["user"].format(
            question=self.format_empty(question),
            answer=self.format_empty(answer)
        )

        raw_output = self._generate_response(sys_prompt, usr_prompt)
        score = self._parse_score(raw_output)

        evaluation_text = raw_output.strip()
        if score == self.default_score:
            evaluation_text += f"\n[WARN] score parsing failed; default score: {self.default_score}"

        return {
            "question": question,
            "model_answer": answer,
            "evaluation_text": evaluation_text,
            "score": score
        }

    def judge_base_batch(self, questions: List[str], answers: List[str]) -> List[Dict]:
        if len(questions) != len(answers):
            raise ValueError("Questions and answers must have the same length")

        results = []
        for q, ans in zip(questions, answers):
            results.append(self.judge_base_single(q, ans))
            self._release_memory()

        return results

    def judge_single(self, question: str, reference: str, model_answer: str) -> Dict:
        sys_prompt = self.prompt_templates["ref_eval"]["system"]
        usr_prompt = self.prompt_templates["ref_eval"]["user"].format(
            question=self.format_empty(question),
            answer_ref=self.format_empty(reference),
            answer_ass=self.format_empty(model_answer)
        )

        raw_output = self._generate_response(sys_prompt, usr_prompt)
        score = self._parse_score(raw_output)

        evaluation_text = raw_output.strip()
        if score == self.default_score:
            evaluation_text += f"\n[WARN] score parsing failed; default score: {self.default_score}"

        return {
            "question": question,
            "reference_answer": reference,
            "model_answer": model_answer,
            "evaluation_text": evaluation_text,
            "score": score
        }

    def judge_batch(self, questions: List[str], references: List[str], model_answers: List[str]) -> List[Dict]:
        if not (len(questions) == len(references) == len(model_answers)):
            raise ValueError("Questions, references, and model answers must have the same length")

        batch_results = []
        for q, ref, ans in zip(questions, references, model_answers):
            batch_results.append(self.judge_single(q, ref, ans))
            self._release_memory(sleep_time=0.3)

        return batch_results
