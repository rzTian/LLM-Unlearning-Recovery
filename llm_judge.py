import torch
import json
import time
import re
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from llm_prompt import EVALUATION_PROMPT, BASE_EVAL_PROMPT  # 初始化时导入模板


class LLMJudge:
    def __init__(self,
                 model_name: str = "deepseek-ai/deepseek-llm-7b-chat",
                 use_quantization: bool = True,
                 max_new_tokens: int = 800):
        """
        初始化LLM评判器（统一Prompt模板绑定+核心参数配置）
        :param model_name: 模型标识（Hugging Face名/本地路径）
        :param use_quantization: 是否启用4-bit量化（节省显存）
        :param max_new_tokens: 生成文本最大长度（适配评估内容）
        """
        # 1. 核心配置参数（集中管理，便于修改）
        self.model_name = model_name
        self.use_quantization = use_quantization
        self.max_new_tokens = max_new_tokens
        self.default_score = -1.0  # 统一错误默认分数：-1.0表示解析失败

        # 2. 绑定外部Prompt模板（初始化时加载，避免函数内重复调用）
        self.prompt_templates = {
            "full_eval": EVALUATION_PROMPT,  # 完整评估（问题+参考答案+模型答案）
            "base_eval": BASE_EVAL_PROMPT    # 基础评估（仅问题+模型答案）
        }

        # 3. 加载Tokenizer和Model（统一初始化流程）
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()

    def _load_tokenizer(self) -> AutoTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, padding_side="right", local_files_only=True
        )
        # 关键修改：设置独立pad_token（优先用模型自带pad_token，无则用<|pad|>）
        if tokenizer.pad_token is None:
            if "<|pad|>" in tokenizer.get_vocab():
                tokenizer.pad_token = "<|pad|>"
            else:
                tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        return tokenizer

    def _load_model(self) -> AutoModelForCausalLM:
        """统一加载模型（支持量化/非量化，复用参数逻辑）"""
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "trust_remote_code": True,
            "local_files_only": True
        }

        # 启用量化配置（仅当use_quantization=True时添加）
        if self.use_quantization:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

        # 核心模型加载（统一参数格式）
        model = AutoModelForCausalLM.from_pretrained(self.model_name,** model_kwargs)
        model.eval()  # 禁用训练模式，节省显存
        return model

    def _parse_score(self, raw_output: str) -> float:
        """优化分数提取：支持 Rating: num / [num] / [[num]] 等多种格式"""
        # 1. 预处理：移除中文、多余空格、空行
        cleaned_output = re.sub(r'[^\x00-\x7F]+', '', raw_output)  # 删除所有中文字符
        cleaned_output = re.sub(r'\s+', ' ', cleaned_output).strip()  # 合并多余空格

        # 2. 正则匹配：支持以下所有格式
        # - Rating: 8.5
        # - Rating: [8.5]
        # - Rating: [[8.5]]
        # - Final Rating: 8.5
        # - Final Rating: [8.5]
        # - Final Rating: [[8.5]]
        # - 甚至缺少前缀的 [[8.5]] 或 [8.5]
        score_pattern = r"(Final\s+)?Rating:\s*(\[\[|\[)?(\d+(\.\d+)?)(\]\]|\])?"
        match = re.search(score_pattern, cleaned_output, re.IGNORECASE)  # 忽略大小写

        if match:
            # 提取核心分数数值（第三组为数字部分）
            score_str = match.group(3)
            try:
                score = round(float(score_str), 1)
                # 验证分数范围（0-10之间为有效分）
                if 0.0 <= score <= 10.0:
                    return score
            except ValueError:
                # 极端情况：匹配到非数字（理论上不会出现）
                return self.default_score
        
        # 无匹配或无效分数，返回默认值
        return self.default_score

    def _release_memory(self, sleep_time: float = 0.2) -> None:
        """辅助函数：释放显存+降低GPU负载（避免重复代码）"""
        torch.cuda.empty_cache()
        time.sleep(sleep_time)

    def _generate_response(self, prompt: str) -> str:
        """通用生成函数（统一对话格式+生成参数，减少重复逻辑）"""
        # 构建DeepSeek标准对话格式
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(self.model.device)

        # 统一生成参数（控制随机性，确保评分稳定）
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=0.2,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # 解码并返回生成结果（去除输入prompt）
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

    # ------------------------------
    # 对外接口1：基础评估（仅问题+答案，返回分数列表）
    # ------------------------------
    def judge_base_single(self, question: str, answer: str) -> Dict:
        """
        单样本基础评估（仅问题+答案）
        :param question: 问题
        :param answer: 模型答案
        :return: 评估结果字典，包含输入信息、评价文本、分数
        """
        # 1. 格式化基础评估Prompt
        prompt = self.prompt_templates["base_eval"].format(
            question=self.format_empty(question),
            answer_model=self.format_empty(answer)
        )

        # 2. 生成评估结果
        raw_output = self._generate_response(prompt)
        score = self._parse_score(raw_output)

        # 3. 补充解析失败标注
        evaluation_text = raw_output.strip()
        if score == self.default_score:
            evaluation_text += f"\n（⚠️  评分解析失败，默认分数：{self.default_score}）"

        # 4. 返回统一格式结果
        return {
            "question": question,
            "model_answer": answer,
            "evaluation_text": evaluation_text,
            "score": score
        }

    def judge_base_batch(self, questions: List[str], answers: List[str]) -> List[Dict]:
        """
        批量基础评估（仅问题+答案）
        :param questions: 问题列表
        :param answers: 模型答案列表
        :return: 评估结果字典列表
        """
        if len(questions) != len(answers):
            raise ValueError("Questions and answers must have the same length")

        results = []
        for q, ans in zip(questions, answers):
            results.append(self.judge_base_single(q, ans))
            self._release_memory()

        return results

    # ------------------------------
    # 对外接口2：完整评估（问题+参考答案+模型答案，返回详细结果）
    # ------------------------------
    def judge_single(self, question: str, reference: str, model_answer: str) -> Dict:
        """单样本完整评估（返回含原文、分数的详细结果）"""
        # 1. 格式化完整评估Prompt
        prompt = self.prompt_templates["full_eval"].format(
            question=self.format_empty(question),
            answer_ref=self.format_empty(reference),
            answer_model=self.format_empty(model_answer)
        )

        # 2. 生成评估结果
        raw_output = self._generate_response(prompt)
        score = self._parse_score(raw_output)

        # 3. 补充解析失败的标注
        evaluation_text = raw_output.strip()
        if score == self.default_score:
            evaluation_text += f"\n（⚠️  评分解析失败，默认分数：{self.default_score}）"

        # 4. 返回统一格式结果
        return {
            "question": question,
            "reference_answer": reference,
            "model_answer": model_answer,
            "evaluation_text": evaluation_text,
            "score": score
        }

    def judge_batch(self, questions: List[str], references: List[str], model_answers: List[str]) -> List[Dict]:
        """批量完整评估（对外接口参数不变，内部复用单样本逻辑）"""
        if not (len(questions) == len(references) == len(model_answers)):
            raise ValueError("Questions, references, and model answers must have the same length")

        batch_results = []
        for q, ref, ans in zip(questions, references, model_answers):
            batch_results.append(self.judge_single(q, ref, ans))
            self._release_memory(sleep_time=0.3)  # 批量评估延长sleep时间

        return batch_results


# ------------------------------
# 使用示例
# ------------------------------
if __name__ == "__main__":
    import json
    MODEL_NAME = "deepseek-ai/deepseek-llm-7b-chat"   # 可替换为"meta-llama/Llama-2-7b-hf"
    print(f"🔍 Testing LLMJudge with model: {MODEL_NAME}\n")

    # 1. 初始化LLM评判器（支持本地缓存，兼容Llama-2/DeepSeek）
    judge = LLMJudge(
        model_name=MODEL_NAME,
        use_quantization=True,  # 4-bit量化（显存不足时必开）
        max_new_tokens=1200  # 足够容纳多语言/长评价
    )
    print("✅ LLM Judge initialized successfully (Local cached model used)\n")

    # 2. 测试数据：覆盖8类核心场景（含异常/边界/多语言）
    test_data = {
        # 基础信息（用于双Batch接口共用）
        "common": {
            "questions": [
                # Case 1: 参考答案正确 + 模型答案完全正确（分数10）
                "What is the chemical symbol for water?",
                # Case 2: 参考答案正确 + 模型答案完全错误（分数0）
                "Which planet is known as the Red Planet?",
                # Case 3: 参考答案错误但含信息 + 模型答案高度贴合（分数9）
                "How many continents are there in the world? (Ref: 6 continents)",
                # Case 4: 参考答案错误但含信息 + 模型答案完全偏离（分数1）
                "What is the boiling point of water? (Ref: 90°C at sea level)",
                # Case 5: 参考答案无有效信息 + 模型答案也无信息（分数5）
                "Tell me about AI. (Ref: I don't know either.)",
                # Case 6: 参考答案无有效信息 + 模型答案正确详细（分数10）
                "Calculate 25 × 4. (Ref: Can you repeat the question?)",
                # Case 7: 异常场景（模型输出空/乱码）
                "What is the capital of Japan?",
                # Case 8: 多语言场景（中文问题+中文答案）
                "中国的首都是哪里？"
            ],
            "answers_model": [
                # Case 1 模型答案（正确）
                "The chemical symbol for water is H₂O, which consists of two hydrogen atoms and one oxygen atom.",
                # Case 2 模型答案（错误）
                "Venus is known as the Red Planet because of its reddish appearance.",
                # Case 3 模型答案（贴合）
                "There are 6 major continents on Earth, excluding Antarctica in some classifications.",
                # Case 4 模型答案（偏离）
                "The freezing point of water is 0°C, which is a common scientific fact.",
                # Case 5 模型答案（无信息）
                "I'm not sure about AI either; it's a broad topic.",
                # Case 6 模型答案（正确详细）
                "25 × 4 = 100. This is a basic multiplication: 25×4=100, 25×8=200, etc.",
                # Case 7 模型答案（空+乱码）
                "",  # 空答案
                # Case 8 模型答案（中文正确）
                "中国的首都是北京，它是中国的政治、文化和国际交流中心。"
            ]
        },
        # 完整评估专属数据（含参考答案+预期分数）
        "full_eval": {
            "answers_ref": [
                # Case 1 参考答案（正确）
                "The chemical symbol for water is H₂O. (Correct reference)",
                # Case 2 参考答案（正确）
                "Mars is known as the Red Planet. (Correct reference)",
                # Case 3 参考答案（错误但含信息）
                "There are 6 continents in the world. (Incorrect but informative: standard is 7)",
                # Case 4 参考答案（错误但含信息）
                "The boiling point of water is 90°C at sea level. (Incorrect: standard is 100°C)",
                # Case 5 参考答案（无有效信息）
                "I don't know either. (No valid info)",
                # Case 6 参考答案（无有效信息）
                "Can you repeat the question? (No valid info)",
                # Case 7 参考答案（正确）
                "The capital of Japan is Tokyo. (Correct reference)",
                # Case 8 参考答案（正确）
                "中国的首都是北京。（正确参考答案）"
            ],
            "expected_scores": [10.0, 0.0, 9.0, 1.0, 5.0, 10.0, 0.0, 10.0]  # 完整评估预期分
        },
        # 基础评估专属数据（仅问题+答案，预期分数）
        "base_eval": {
            "expected_scores": [10.0, 0.0, 8.5, 2.0, 3.0, 10.0, 0.0, 10.0]  # 基础评估预期分（相关性+准确性平均）
        }
    }

    # ------------------------------
    # 测试1：judge_batch（完整评估，含参考答案）
    # ------------------------------
    print("="*85)
    print("🚀 Test 1: judge_batch (Full Evaluation - Question + Reference + Model Answer)")
    print("="*85)
    full_results = judge.judge_batch(
        questions=test_data["common"]["questions"],
        references=test_data["full_eval"]["answers_ref"],
        model_answers=test_data["common"]["answers_model"]
    )

    # 统计完整评估结果（允许±2.0误差，LLM评分存在合理波动）
    full_pass_count = 0
    full_details = []
    for idx, (res, exp) in enumerate(zip(full_results, test_data["full_eval"]["expected_scores"]), 1):
        # 判断是否通过（误差≤2.0）
        is_pass = abs(res["score"] - exp) <= 2.0
        if is_pass:
            full_pass_count += 1
        # 收集详情（用于后续打印）
        full_details.append({
            "case": idx,
            "is_pass": is_pass,
            "question": res["question"],
            "ref_answer": res["reference_answer"],
            "model_answer": res["model_answer"],
            "actual_score": res["score"],
            "expected_score": exp,
            "eval_text": res["evaluation_text"]
        })

    # 打印完整评估详情
    for detail in full_details:
        print(f"\n📌 Case {detail['case']}: {'PASS' if detail['is_pass'] else 'FAIL'}")
        print(f"   Question: {detail['question']}")
        print(f"   Ref Answer: {detail['ref_answer'][:80]}..." if len(detail['ref_answer']) > 80 else f"   Ref Answer: {detail['ref_answer']}")
        print(f"   Model Answer: {detail['model_answer'][:80]}..." if len(detail['model_answer']) > 80 else f"   Model Answer: {detail['model_answer']}")
        print(f"   Score: {detail['actual_score']} (Expected: {detail['expected_score']})")
        print(f"   Eval Text: \n {detail['eval_text']}")

    # 打印完整评估统计
    full_pass_rate = (full_pass_count / len(full_results)) * 100
    print(f"\n📊 Full Evaluation Summary: {full_pass_count}/{len(full_results)} Cases Passed ({full_pass_rate:.1f}%)")

    # ------------------------------
    # 测试2：judge_base_batch（基础评估，仅问题+答案）
    # ------------------------------
    print("\n" + "="*85)
    print("🚀 Test 2: judge_base_batch (Base Evaluation - Only Question + Model Answer)")
    print("="*85)
    base_results = judge.judge_base_batch(
        questions=test_data["common"]["questions"],
        answers=test_data["common"]["answers_model"]
    )

    # 统计基础评估结果（允许±3.0误差，基础评估维度更少）
    base_pass_count = 0
    base_details = []
    for idx, (res, exp) in enumerate(zip(base_results, test_data["base_eval"]["expected_scores"]), 1):
        # 判断是否通过（误差≤1.0）
        is_pass = abs(res["score"] - exp) <= 3.0
        if is_pass:
            base_pass_count += 1
        # 收集详情
        base_details.append({
            "case": idx,
            "is_pass": is_pass,
            "question": res["question"],
            "model_answer": res["model_answer"],
            "actual_score": res["score"],
            "expected_score": exp,
            "eval_text": res["evaluation_text"]
        })

    # 打印基础评估详情
    for detail in base_details:
        print(f"\n📌 Case {detail['case']}: {'PASS' if detail['is_pass'] else 'FAIL'}")
        print(f"   Question: {detail['question']}")
        print(f"   Model Answer: {detail['model_answer'][:80]}..." if len(detail['model_answer']) > 80 else f"   Model Answer: {detail['model_answer']}")
        print(f"   Score: {detail['actual_score']} (Expected: {detail['expected_score']})")
        print(f"   Eval Text: \n {detail['eval_text']}")

    # 打印基础评估统计
    base_pass_rate = (base_pass_count / len(base_results)) * 100
    print(f"\n📊 Base Evaluation Summary: {base_pass_count}/{len(base_results)} Cases Passed ({base_pass_rate:.1f}%)")

    # ------------------------------
    # 总体测试总结
    # ------------------------------
    print("\n" + "="*85)
    print("🏆 Overall Test Summary (Both Batch Interfaces)")
    print("="*85)
    print(f"1. judge_batch (Full Evaluation): {full_pass_count}/{len(full_results)} Passed ({full_pass_rate:.1f}%)")
    print(f"2. judge_base_batch (Base Evaluation): {base_pass_count}/{len(base_results)} Passed ({base_pass_rate:.1f}%)")
    print(f"3. Total Pass Rate: {full_pass_count + base_pass_count}/{len(full_results) + len(base_results)} Cases Passed ({((full_pass_count + base_pass_count) / (len(full_results) + len(base_results)))*100:.1f}%)")

    # 可选：保存测试结果到JSON文件（便于后续分析）
    save_path = f"llm_judge.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({
            "test_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "model_name": judge.model_name,
            "full_evaluation": full_details,
            "base_evaluation": base_details,
            "summary": {
                "full_pass_count": full_pass_count,
                "full_total_count": len(full_results),
                "full_pass_rate": full_pass_rate,
                "base_pass_count": base_pass_count,
                "base_total_count": len(base_results),
                "base_pass_rate": base_pass_rate,
                "total_pass_rate": ((full_pass_count + base_pass_count) / (len(full_results) + len(base_results)))*100
            }
        }, f, ensure_ascii=False, indent=4)
    print(f"\n💾 Test results saved to: {save_path}")