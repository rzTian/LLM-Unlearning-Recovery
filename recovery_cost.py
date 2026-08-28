import json
import torch
import math
import time
import statistics
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, LogitsProcessorList
from peft import PeftModel, LoraConfig, get_peft_model
import torch.nn.functional as F
from typing import List, Tuple
import os
import logging
import argparse
from contextlib import contextmanager
# For computing the edit distance
import re
import Levenshtein

from argsetting import parser_eval
from evaluate import EvalQA
from utils import CustomizedLogitsProcessor, compute_dpo_loss

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


class CostMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.num_samples = 0
        self.total_target_tokens = 0
        self.forward_passes = 0
        self.backward_passes = 0
        self.candidate_sizes = []
        self.beam_active_sizes = []
        self.beam_expansions = []
        self.start_time = None
        self.end_time = None
        self.decode_elapsed_sec = 0.0
        self._decode_start_time = None
        self.timing_sec = {
            "forward": 0.0,
            "candidate_lookup": 0.0,
            "logits_selection": 0.0,
            "state_update": 0.0,
        }
        self._cuda_timing_events = {name: [] for name in self.timing_sec}
        self.peak_allocated_gb = None
        self.peak_reserved_gb = None

    def start(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.start_time = time.perf_counter()
        self.end_time = None

    def stop(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.end_time = time.perf_counter()
        if torch.cuda.is_available():
            self.peak_allocated_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            self.peak_reserved_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
        else:
            self.peak_allocated_gb = None
            self.peak_reserved_gb = None

    def add_sample(self, target_len: int):
        self.num_samples += 1
        self.total_target_tokens += int(target_len)

    def add_forward(self, n: int = 1):
        self.forward_passes += int(n)

    def add_candidate_size(self, n: int):
        self.candidate_sizes.append(int(n))

    def add_beam_active(self, n: int):
        self.beam_active_sizes.append(int(n))

    def add_beam_expansion(self, n: int):
        self.beam_expansions.append(int(n))

    @staticmethod
    def _sync_cuda():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def begin_decode(self):
        self._sync_cuda()
        self._decode_start_time = time.perf_counter()

    def end_decode(self):
        self._sync_cuda()
        if self._decode_start_time is not None:
            self.decode_elapsed_sec += time.perf_counter() - self._decode_start_time
        self._decode_start_time = None
        self._flush_cuda_timing_events()

    def _flush_cuda_timing_events(self):
        if not torch.cuda.is_available():
            return
        for name, pairs in self._cuda_timing_events.items():
            for start_event, end_event in pairs:
                self.timing_sec[name] = self.timing_sec.get(name, 0.0) + start_event.elapsed_time(end_event) / 1000.0
            pairs.clear()

    @contextmanager
    def timed(self, name):
        if torch.cuda.is_available() and name in {"forward", "logits_selection"}:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            try:
                yield
            finally:
                end_event.record()
                self._cuda_timing_events.setdefault(name, []).append((start_event, end_event))
        else:
            t0 = time.perf_counter()
            try:
                yield
            finally:
                self.timing_sec[name] = self.timing_sec.get(name, 0.0) + (time.perf_counter() - t0)

    @staticmethod
    def _mean(values):
        return float(statistics.mean(values)) if values else None

    @staticmethod
    def _median(values):
        return float(statistics.median(values)) if values else None

    @staticmethod
    def _min(values):
        return int(min(values)) if values else None

    @staticmethod
    def _max(values):
        return int(max(values)) if values else None

    @staticmethod
    def _method_name(recover_type, flip):
        if recover_type == "hf_greedy":
            return "HF Greedy"
        if recover_type in ("greedy", "manual_greedy"):
            return "Manual Greedy"
        if recover_type == "flip":
            return "RG" if int(flip) == 1 else "RIG"
        if recover_type == "beam":
            return "Beam-RG" if int(flip) == 1 else "Beam-RIG"
        return recover_type

    @staticmethod
    def _impl_name(recover_type):
        if recover_type == "hf_greedy":
            return "HF generate"
        if recover_type in ("greedy", "manual_greedy"):
            return "manual loop"
        if recover_type == "flip":
            return "manual loop + restricted logits"
        if recover_type == "beam":
            return "manual beam + restricted logits"
        return "manual loop"

    def summary(self, eval_args, recover_type, flip, K, C, N):
        self._flush_cuda_timing_events()
        elapsed = self.decode_elapsed_sec
        if elapsed <= 0 and self.start_time is not None and self.end_time is not None:
            elapsed = self.end_time - self.start_time

        avg_target_len = (
            self.total_target_tokens / self.num_samples
            if self.num_samples else 0.0
        )
        forward_per_target_token = (
            self.forward_passes / self.total_target_tokens
            if self.total_target_tokens else 0.0
        )
        sec_per_sample = (
            elapsed / self.num_samples
            if elapsed is not None and self.num_samples else None
        )
        ms_per_target_token = (
            1000.0 * elapsed / self.total_target_tokens
            if elapsed is not None and self.total_target_tokens else None
        )
        ms_per_forward = (
            1000.0 * elapsed / self.forward_passes
            if elapsed is not None and self.forward_passes else None
        )
        tokens_per_sec = (
            self.total_target_tokens / elapsed
            if elapsed is not None and elapsed > 0 else None
        )

        gpu_name = None
        if torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name()
            except Exception:
                gpu_name = None

        epoch = (
            eval_args.eps_fgt
            if eval_args.modelType in ['unlearned', 'pt-unlearned']
            else eval_args.epochs
        )

        return {
            "cost_version": "recovery_cost_v1",
            "method_name": self._method_name(recover_type, flip),
            "impl_path": self._impl_name(recover_type),
            "recover_type": recover_type,
            "flip": None if recover_type in ("hf_greedy", "greedy", "manual_greedy") else int(flip),
            "K": int(K),
            "C": int(C),
            "N": int(N),
            "model_name": eval_args.model_name,
            "model_type": eval_args.modelType,
            "unlearn_method": eval_args.unlearn_method,
            "unlearn_set": eval_args.unlearnSet,
            "dataset_type": eval_args.datasetType,
            "epoch": int(epoch),
            "lr_fgt": float(eval_args.lr_fgt),
            "quant": eval_args.quant,
            "num_samples": int(self.num_samples),
            "total_target_tokens": int(self.total_target_tokens),
            "avg_target_len": avg_target_len,
            "forward_passes": int(self.forward_passes),
            "forward_per_target_token": forward_per_target_token,
            "backward_passes": int(self.backward_passes),
            "candidate_size_mean": self._mean(self.candidate_sizes),
            "candidate_size_median": self._median(self.candidate_sizes),
            "candidate_size_min": self._min(self.candidate_sizes),
            "candidate_size_max": self._max(self.candidate_sizes),
            "beam_active_mean": self._mean(self.beam_active_sizes),
            "beam_active_max": self._max(self.beam_active_sizes),
            "beam_expansion_mean": self._mean(self.beam_expansions),
            "total_wall_time_sec": elapsed,
            "sec_per_sample": sec_per_sample,
            "ms_per_target_token": ms_per_target_token,
            "ms_per_forward": ms_per_forward,
            "forward_ms_per_target_token": 1000.0 * self.timing_sec.get("forward", 0.0) / self.total_target_tokens if self.total_target_tokens else None,
            "candidate_lookup_ms_per_target_token": 1000.0 * self.timing_sec.get("candidate_lookup", 0.0) / self.total_target_tokens if self.total_target_tokens else None,
            "logits_selection_ms_per_target_token": 1000.0 * self.timing_sec.get("logits_selection", 0.0) / self.total_target_tokens if self.total_target_tokens else None,
            "state_update_ms_per_target_token": 1000.0 * self.timing_sec.get("state_update", 0.0) / self.total_target_tokens if self.total_target_tokens else None,
            "tokens_per_sec": tokens_per_sec,
            "peak_allocated_gb": self.peak_allocated_gb,
            "peak_reserved_gb": self.peak_reserved_gb,
            "gpu_name": gpu_name,
            "cuda_available": torch.cuda.is_available(),
            "model_loading_excluded": True,
        }


class recoverQA(EvalQA):
    
    def __init__(self, recover_type, recover_mode='greedy', flip=1, loss_type='ce', beta=1.0,
                 K=1, C=1, N=1, entro=False, profile_debug=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recover_type = recover_type.lower().strip()  # 统一小写
        self.recover_mode = recover_mode if recover_mode in ('greedy', 'oracle') else 'greedy'  # 目前仅用于 grad-recover
        self.flip = int(flip)  # CustomizedLogitsProcessor 中 1=取负号后 argmax，等价选最小原始 logit
        self.loss_type = loss_type.lower().strip()  # 'ce' 或 'npo'
        self.beta = float(beta)  # NPO 权重参数
        self.K = int(K)
        self.C = int(C)
        self.N = int(N)
        self.entro = entro
        self.profile_debug = profile_debug
        self.cost = CostMeter()
        self._candidate_tensor_cache = {}
        self._candidate_processor = None

    def _ensure_pad_token(self):
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                try:
                    self.model.resize_token_embeddings(len(self.tokenizer))
                except Exception:
                    pass

    def _digit_token_ids(self):
        # 仅作兜底（当某一位没有收集到候选集时）
        ids = []
        for d in range(10):
            toks = self.tokenizer.encode(str(d), add_special_tokens=False)
            if len(toks) != 1:
                print(f"[WARN] digit '{d}' tokenized to {toks}, using first token {toks[0]}")
            ids.append(toks[0])
        return ids

    def _concat_inputs(self, base_ids, extra_ids):
        device = base_ids.device
        if isinstance(extra_ids, list):
            extra_ids = torch.tensor([extra_ids], dtype=torch.long, device=device)
        elif isinstance(extra_ids, torch.Tensor):
            if extra_ids.dim() == 1:
                extra_ids = extra_ids.unsqueeze(0)
            extra_ids = extra_ids.to(device)
        else:
            raise ValueError("extra_ids must be list[int] or Tensor")
        input_ids = torch.cat([base_ids, extra_ids], dim=-1)
        attn = torch.ones_like(input_ids, device=device)
        return input_ids, attn

    def _safe_decode(self, token_id):
        try:
            return self.tokenizer.decode([token_id])
        except Exception:
            return f"<{token_id}>"

    def _candidate_ids_for(self, attr_type, step, history, device):
        cache_key = (str(device), attr_type, int(step), tuple(int(x) for x in history))
        if cache_key in self._candidate_tensor_cache:
            return self._candidate_tensor_cache[cache_key]

        if self._candidate_processor is None:
            self._candidate_processor = CustomizedLogitsProcessor(
                tokenizer=self.tokenizer,
                attr_type=attr_type,
                generation_step=0,
                flip_logit=self.flip,
            )

        key = f"{attr_type}_pos{step}"
        selected = self._candidate_processor.token_sets.get(key, [self.tokenizer.eos_token_id])
        selected = list(selected) if isinstance(selected, set) else list(selected)

        rules = self._candidate_processor.dependency_rules
        if attr_type in rules and step in rules[attr_type]:
            for dep_pos, dep_map in rules[attr_type][step].items():
                if dep_pos < len(history):
                    hist_token = int(history[dep_pos])
                    if hist_token in dep_map:
                        allowed = set(int(x) for x in dep_map[hist_token])
                        narrowed = [int(x) for x in selected if int(x) in allowed]
                        if narrowed:
                            selected = narrowed

        tensor = torch.as_tensor(selected, device=device, dtype=torch.long)
        self._candidate_tensor_cache[cache_key] = tensor
        return tensor

    def _decode_with_manual_loop(self, questions, output_length, attr_type=None, answer=None, restricted=False):
        inputs = self.tokenizer(
            questions,
            padding=True,
            truncation=True,
            max_length=150,
            return_tensors="pt"
        ).to(self.device)

        max_steps = output_length[0] if output_length else 20
        outputs = []
        generated_history = []
        orders_all = [[]]
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        next_input_ids = input_ids
        past_key_values = None

        answer_ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"] if answer else []

        self.cost.begin_decode()
        with torch.inference_mode():
            for step in range(max_steps):
                with self.cost.timed("forward"):
                    out = self.model(
                        input_ids=next_input_ids,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        return_dict=True,
                        use_cache=True,
                    )
                self.cost.add_forward(1)
                past_key_values = out.past_key_values
                logits_step = out.logits[:, -1, :]

                if restricted:
                    with self.cost.timed("candidate_lookup"):
                        candidate_ids = self._candidate_ids_for(attr_type, step, generated_history, logits_step.device)
                    self.cost.add_candidate_size(int(candidate_ids.numel()))
                    with self.cost.timed("logits_selection"):
                        vals = logits_step.index_select(-1, candidate_ids)
                        idx = torch.argmin(vals, dim=-1) if self.flip == 1 else torch.argmax(vals, dim=-1)
                        next_token = candidate_ids[idx].view(1, 1)
                else:
                    self.cost.add_candidate_size(logits_step.size(-1))
                    with self.cost.timed("logits_selection"):
                        next_token = torch.argmax(logits_step, dim=-1, keepdim=True)
                    candidate_ids = None

                with self.cost.timed("state_update"):
                    current_token = int(next_token[0, 0].item())
                    generated_history.append(current_token)
                    outputs.append(next_token)
                    next_input_ids = next_token
                    attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)

                if restricted and self.profile_debug and step < len(answer_ids) and candidate_ids is not None:
                    rank = self.get_token_rank(logits_step[0], answer_ids[step], candidate_ids.tolist(), verbose=False)
                    orders_all[0].append(f"{rank}/{int(candidate_ids.numel()) - 1}")

        self.cost.end_decode()

        full_output_ids = torch.cat(outputs, dim=1)
        decoded_outputs = self.tokenizer.batch_decode(full_output_ids, skip_special_tokens=False)
        return [(decoded_outputs[0], orders_all[0])]

    def recover_by_hf_greedy(self, questions, output_length, attr_type, answer):
        inputs = self.tokenizer(
            questions,
            padding=True,
            truncation=True,
            max_length=150,
            return_tensors="pt"
        ).to(self.device)

        max_steps = output_length[0] if output_length else 20
        self.cost.begin_decode()
        with torch.inference_mode():
            with self.cost.timed("forward"):
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_steps,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
        self.cost.end_decode()
        self.cost.add_forward(max_steps)
        self.cost.add_candidate_size(self.model.config.vocab_size)

        prompt_len = inputs["input_ids"].shape[-1]
        full_output_ids = output[:, prompt_len:prompt_len + max_steps]
        decoded_outputs = self.tokenizer.batch_decode(full_output_ids, skip_special_tokens=False)
        return [(decoded_outputs[0], [])]

    def _grad_norms_last_layer(self, logits_last, hidden_last, cand_ids, ref_logits_last=None):
        """
        用闭式公式计算最后一层权重 W 的梯度范数（针对给定候选 token 集）。
        - CE:   ||∇_W CE||_F = ||(p - e_y)||_2 * ||h||_2
        - NPO:  使用 DPO 形式的“单正样本”缩放，
                L_y = -log σ(β·Δ_y)，Δ_y = log π(y) - log π_ref(y)。
                则 ||∇_W L_y||_F = β·σ(-β·Δ_y)·|| (e_y - p) ||_2 · ||h||_2
        """
        import torch
        device = logits_last.device
        # probs
        p = torch.softmax(logits_last, dim=-1).squeeze(0)  # [V]
        p_sq_sum = torch.sum(p * p)                        # 标量
        h = hidden_last.squeeze(0).squeeze(0)              # [H]
        h_norm = torch.norm(h, p=2) + 1e-12

        # 如需 NPO 权重，获取 q
        use_npo = (self.loss_type == 'npo')
        if use_npo:
            if ref_logits_last is None:
                # 计算 ref logits
                self.ref_model.eval()
                ref_out = self.ref_model(
                    input_ids=self._last_input_ids,            # 下面 recover_by_grad 会设置这个缓存
                    attention_mask=self._last_attn_mask,
                    output_hidden_states=False,
                    use_cache=False
                )
                ref_logits_last = ref_out.logits[:, -1, :].float()
            q = torch.softmax(ref_logits_last, dim=-1).squeeze(0)

        results = []
        beta = torch.tensor(self.beta, device=device, dtype=p.dtype)
        eps = 1e-20

        for tid in cand_ids:
            tid = int(tid)
            p_t = p[tid].clamp_min(eps)
            # CE 的 ||p - e_t||_2
            ce_vec_norm = torch.sqrt(1.0 - 2.0 * p_t + p_sq_sum + 1e-12)
            scale = 1.0
            if use_npo:
                q_t = q[tid].clamp_min(eps)
                # W = 2 p_t^β / (p_t^β + q_t^β)
                num = 2.0 * torch.pow(p_t, beta)
                den = torch.pow(p_t, beta) + torch.pow(q_t, beta)
                scale = (num / den).detach()  # 不回传梯度
            g_norm = (scale * ce_vec_norm * h_norm).item()
            results.append((tid, g_norm))
        return results

    def _should_stop_attr(self, attr_type, out_tok_ids, new_tok_id):
        # 1) 通用句号停止
        if new_tok_id == 13:
            return True

        # 2) 基于解码文本的属性级停止
        text = self.tokenizer.decode(out_tok_ids, skip_special_tokens=False)

        if attr_type == "year_of_birth":
            # 例如 " 1987."
            return re.fullmatch(r"\s*\d{4}\.", text) is not None

        elif attr_type == "address_postcode":
            # 例如 " A1B2C3."
            return re.fullmatch(r"\s*[A-Z]\d[A-Z]\d[A-Z]\d\.", text) is not None

        elif attr_type == "social_insurance_number":
            digits = re.sub(r"\D", "", text)
            return len(digits) >= 9 or text.endswith(".")

        elif attr_type == "blood_type":
            # blood_type 一般两步就够，或者直接用字符串集合判断
            return text.strip() in {"A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"} \
                or text.strip().endswith(".")
        
        return False

    def get_token_rank(self, logits, token_id, selected_token_ids, step=None, verbose=False):
        """
        Compute and print the rank of token_id among selected_token_ids, highlighting:
        ✔ = answer token, ★ = top-1 logit token
        """
        if not selected_token_ids:
            return -1

        sel_ids = torch.as_tensor(selected_token_ids, device=logits.device, dtype=torch.long)
        candidate_logits = logits.index_select(0, sel_ids).detach()
        sorted_indices = torch.argsort(candidate_logits, descending=True)

        # Find top-1 token and answer index
        token_id = int(token_id)
        matches = (sel_ids == token_id).nonzero(as_tuple=True)[0]
        idx_in_candidates = int(matches[0].item()) if matches.numel() else -1
        top1_idx = int(sorted_indices[0].item())

        # Header
        if verbose:
            if step is not None:
                print(f"\n[Step {step}] selected_token_ids and logits:")
            else:
                print("\nSelected token logits:")

            # Print with markers only for explicit debugging; this is expensive in timing runs.
            candidate_logits_cpu = candidate_logits.cpu()
            selected_token_ids_cpu = sel_ids.cpu().tolist()
            for idx, tid in enumerate(selected_token_ids_cpu):
                token_str = self.tokenizer.decode([tid])
                logit_val = candidate_logits_cpu[idx].item()

                markers = ""
                if idx == idx_in_candidates:
                    markers += "✔"
                if idx == top1_idx:
                    markers += "★"

                print(f"{markers:2} {repr(token_str):>6} (id={tid:>5}): {logit_val:.4f}")

        # Rank of the answer
        if idx_in_candidates != -1:
            rank = int((sorted_indices == idx_in_candidates).nonzero(as_tuple=True)[0].item())
        else:
            rank = -1

        if verbose:
            print(f"Answer rank: {rank}")
        return rank

    def recover_by_greedy(self, questions, output_length, attr_type, answer):
        return self._decode_with_manual_loop(questions, output_length, attr_type, answer, restricted=False)

    def recover_by_flip(self, questions, output_length, attr_type, answer):
        assert attr_type is not None, "attr_type must be provided."
        return self._decode_with_manual_loop(questions, output_length, attr_type, answer, restricted=True)


    def recover_by_beam(self, questions, output_length, attr_type, answer, K=5, C=3):
        """
        扩展 beam-search 的恢复算法（复杂度 O(K*C*n)）。
        - questions: List[str]，长度通常为1（按你现有评测一条条做）
        - output_length: List[int]，目标输出长度（例如 [6] 表示预测 6 位）
        - attr_type: 属性类型（沿用你 CustomizedLogitsProcessor 的分支）
        - answer: 可选，用于对齐/观测rank；不用于搜索逻辑
        - K: 每轮保留的序列条数
        - C: 每个序列拓展的分支数
        返回：[(decoded_str, score, token_id_list)] 按 score 降序的前 K 条
        """
        assert attr_type is not None, "attr_type must be provided."
        assert questions and isinstance(questions, (list, tuple)), "questions 必须是非空列表/元组"
        max_steps = output_length[0] if output_length else 20

        # ========== 编码输入 ==========
        inputs = self.tokenizer(
            questions,
            padding=True,
            truncation=True,
            max_length=150,
            return_tensors="pt",
        ).to(self.device)

        # 初始 beam：元素为 (log_score, input_ids, attention_mask, out_token_ids)
        beams = [(
            0.0,
            inputs["input_ids"].clone(),
            inputs["attention_mask"].clone(),
            []
        )]

        # ========== 逐位扩展 ==========
        self.cost.begin_decode()
        with torch.inference_mode():
            finished_beams = []

            for step in range(max_steps):
                self.cost.add_beam_active(len(beams))
                new_beams = []

                for log_score, input_ids, attention_mask, out_tok_ids in beams:
                    # 构造 processor（带 flip 与属性约束）
                    processor = CustomizedLogitsProcessor(
                        tokenizer=self.tokenizer,
                        attr_type=attr_type,
                        generation_step=step,
                        flip_logit=self.flip,  # 0/1
                    )
                    processor.history = out_tok_ids
                    lp_list = LogitsProcessorList([processor])

                    # 前向计算 logits（取最后一位）
                    with self.cost.timed("forward"):
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            return_dict=True
                        )
                    self.cost.add_forward(1)
                    next_logits = outputs.logits[:, -1, :]  # shape: [1, vocab]

                    # 应用 processor（HF 约定：传入 input_ids 与 logits）
                    with self.cost.timed("logits_selection"):
                        next_logits = lp_list(input_ids, next_logits)[0]  # -> shape: [vocab]

                    # 只考虑 processor 指定的候选集合；若 processor 未给出，则用全词表
                    with self.cost.timed("candidate_lookup"):
                        sel_ids = getattr(processor, "selected_token_ids", None)
                        if not sel_ids:
                            sel_ids = torch.arange(next_logits.size(-1), device=next_logits.device, dtype=torch.long)
                        else:
                            # 关键：统一成 LongTensor（避免后续索引返回 python int）
                            sel_ids = torch.as_tensor(list(sel_ids), device=next_logits.device, dtype=torch.long)
                    self.cost.add_candidate_size(int(sel_ids.numel()))

                    with self.cost.timed("logits_selection"):
                        sel_logits = next_logits[sel_ids]  # [num_selected]
                        scores = sel_logits
                        # 用 log-softmax 作为“可累计”的路径分数（不会饱和/下溢）
                        log_probs = torch.log_softmax(sel_logits, dim=-1)   # [M]
                        probs = torch.exp(log_probs)                        # [M]

                    # 归一化熵 -> 自适应权重 alpha ∈ [0.1,1]
                    alpha = 1.0
                    if self.entro:  # entro控制alpha是否生效
                        H = -(probs * log_probs).sum()
                        H_max = math.log(probs.numel()) if probs.numel() > 0 else 1.0
                        alpha = 1.0 - float(H / H_max) if H_max > 0 else 1.0
                        alpha = max(0.1, min(1.0, alpha))
                        print(f'H:{H}, Hmax:{H_max}, alpha:{alpha}')

                    # 选 Top-C
                    topC = min(C, scores.numel())
                    self.cost.add_beam_expansion(topC)
                    with self.cost.timed("logits_selection"):
                        top_scores, top_idx = torch.topk(scores, k=topC, largest=True, sorted=True)

                    with self.cost.timed("state_update"):
                        for j in range(topC):
                            # 现在 sel_ids[...] 是 0-dim Tensor，可以安全 .item()
                            idx = top_idx[j]
                            tok_id = sel_ids[idx].item()
                            incr = float(scores[idx]) # float(log_probs[idx])
                            new_log_score = log_score + alpha * incr
                            new_out_tok_ids = out_tok_ids + [tok_id]

                            tok = torch.tensor([[tok_id]], device=input_ids.device, dtype=input_ids.dtype)
                            new_input_ids = torch.cat([input_ids, tok], dim=-1)
                            new_attention_mask = torch.cat([attention_mask, torch.ones_like(tok)], dim=-1)

                            if self._should_stop_attr(attr_type, new_out_tok_ids, tok_id):
                                finished_beams.append((new_log_score, new_input_ids, new_attention_mask, new_out_tok_ids))
                            else:
                                new_beams.append((new_log_score, new_input_ids, new_attention_mask, new_out_tok_ids))

                # 若没有候选（过度约束），提前结束
                if not new_beams:
                    break

                # 全局排序，保留 K
                new_beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:K]
                beams = new_beams
            
            if finished_beams:
                beams = sorted(finished_beams, key=lambda x: x[0], reverse=True)[:K]
        self.cost.end_decode()

        # ========== 汇总结果 ==========
        results = []
        for log_score, _input_ids, _attn, out_tok_ids in beams:
            decoded = self.tokenizer.decode(out_tok_ids, skip_special_tokens=True)
            results.append((decoded, log_score, out_tok_ids))

        results.sort(key=lambda x: x[1], reverse=True)
        return results
    

    def recover_by_grad(self, questions, output_length, attr_type, answer):
        """
        用 CustomizedLogitsProcessor 提供的 per-position 候选 token 集，
        按 R2:  y_i = arg{max/min}_{y∈候选} ||∇_w CE(f(X,前缀), y)|| 逐位恢复。
        - recover_mode='greedy' ：每步把“选择出来的 y_i”接回 prompt（可能错）
        - recover_mode='oracle' ：每步把“真值的 y_i”接回 prompt（teacher forcing）

        flip 规则：
        - self.flip == 1  -> 选梯度最大的 y_i
        - self.flip == 0  -> 选梯度最小的 y_i

        新增：
        - loss_type: 'ce' 或 'npo'（默认沿用 self.loss_type）
        - beta: 覆盖 self.beta（仅 NPO）
        - beam_size: KC，>=1；1 时退化为原来的 greedy/oracle

        返回：
        List[Tuple[str, List[str]]]
        其中每个元素是 (生成文本, orders)，orders 为每步“真值 token 在候选中的梯度排名”字符串，如 "3/9"
        """
        device = self.model.device
        self._ensure_pad_token()

        if self.loss_type == 'npo' and self.ref_model is None:
            print("[WARN] loss_type='npo' 但未提供 ref_model，自动回退到 'ce'。")
            self.loss_type = 'ce'

        # 目标长度
        L = int(output_length[0]) if isinstance(output_length, (list, tuple)) else int(output_length)
        L = max(L, 1)

        # 真值 token 序列（用于 oracle & 计算真值排名）
        answer_ids_full = self.tokenizer(answer, add_special_tokens=False)["input_ids"]
        if len(answer_ids_full) == 0:
            print(f"[WARN] 空答案 '{answer}'，oracle 模式下将无法 teacher forcing。")
        if len(answer_ids_full) < L:
            L = len(answer_ids_full) if len(answer_ids_full) > 0 else L

        results = []

        for q in questions:
            enc = self.tokenizer(q, return_tensors="pt", add_special_tokens=False)
            base_ids = enc["input_ids"].to(device)
            base_attn = torch.ones_like(base_ids, device=device)

            # beam = [(prefix_tokens:list, orders_step:list[str], score:float)]
            K = int(getattr(self, "K", 1))   # 全局 beam 宽度（保留的路径数）
            C = int(getattr(self, "C", 1))   # 每步在候选 token 中保留的分支数
            beams = [([], [], 0.0)]

            for step in range(L):
                # —— 候选 token 集（来自 CustomizedLogitsProcessor.attr_token_sets）——
                processor = CustomizedLogitsProcessor(
                    tokenizer=self.tokenizer,
                    attr_type=attr_type,
                    generation_step=step,
                    flip_logit=self.flip
                )
                key = f"{attr_type}_pos{step}"
                selected = processor.token_sets.get(key, None)
                if isinstance(selected, set):
                    selected = list(selected)
                if not selected or len(selected) == 0:
                    if step == 0:
                        print(f"[WARN] 未找到 {key} 的候选集，退化为 digits。")
                    cand_ids = self._digit_token_ids()
                else:
                    cand_ids = sorted(set(int(x) for x in selected))

                true_id = int(answer_ids_full[step]) if step < len(answer_ids_full) else None
                choose_max = (self.flip == 1)
                new_beams = []

                # 仅对当前前 K 条 beam 进行扩展（避免指数爆炸）
                # 这里 beams 已经是上一步保留下来的 top-K
                for prefix_tokens, orders_step, acc_score in beams:
                    # —— 组装当前输入 —— 
                    if len(prefix_tokens) > 0:
                        input_ids, attn = self._concat_inputs(base_ids, prefix_tokens)
                    else:
                        input_ids, attn = base_ids, base_attn

                    # —— 一次前向（主模型）/（可选）参考模型 —— 
                    self.model.eval()
                    out = self.model(
                        input_ids=input_ids,
                        attention_mask=attn,
                        output_hidden_states=True,
                        use_cache=False
                    )
                    logits_last = out.logits[:, -1, :].float()             # [1, V]
                    hidden_last = out.hidden_states[-1][:, -1, :].float()  # [1, H]

                    # 为 _grad_norms_last_layer 提供当前输入（以便它需要时前向 ref）
                    self._last_input_ids = input_ids
                    self._last_attn_mask = attn

                    ref_logits_last = None
                    if self.loss_type == 'npo':
                        self.ref_model.eval()
                        ref_out = self.ref_model(
                            input_ids=input_ids,
                            attention_mask=attn,
                            output_hidden_states=False,
                            use_cache=False
                        )
                        ref_logits_last = ref_out.logits[:, -1, :].float()

                    # —— 计算每个候选的梯度范数 —— 
                    norms = self._grad_norms_last_layer(logits_last, hidden_last, cand_ids, ref_logits_last)

                    # —— 排序与打印 —— 
                    pairs = sorted(norms, key=lambda x: x[1], reverse=choose_max)
                    arrow = "↑max" if choose_max else "↓min"
                    lt = self.loss_type.upper()
                    print(f"\n[Grad-{self.recover_mode}|{lt}|beam≤{K}, C={C}] step={step+1}/{L} {arrow}")
                    for tid, g in pairs:
                        mark = []
                        if true_id is not None and tid == true_id: mark.append("<true>")
                        toks = self._safe_decode(tid)
                        print(f"{repr(toks)}:{tid:>6}  -> {g:.6f} {' '.join(mark)}")

                    # —— beam 扩展：每个 beam 仅取前 C 个候选分支 —— 
                    take = min(C, len(pairs))
                    top_c = pairs[:take]  # 已按 choose_max 排序，无需再处理
                    for tid, g in top_c:
                        new_prefix = list(prefix_tokens)
                        if self.recover_mode == 'greedy':
                            new_prefix.append(int(tid))
                        else:  # oracle
                            if true_id is not None:
                                new_prefix.append(int(true_id))

                        # 记录真值排名
                        rank_list = [tid_ for tid_, _ in pairs]
                        if true_id in rank_list:
                            orders_str = f"{rank_list.index(true_id)}/{len(rank_list)-1}"
                        else:
                            orders_str = f"NA/{len(rank_list)-1}"
                        new_orders = list(orders_step) + [orders_str]

                        # 累计分数：最大化取 +g；最小化取 -g（把目标统一为“越大越好”）
                        new_score = acc_score + (g if choose_max else -g)

                        new_beams.append((new_prefix, new_orders, new_score))

                # —— 保留全局前 K 条 beam（按累计分数排序）—— 
                new_beams.sort(key=lambda x: x[2], reverse=True)
                beams = new_beams[:K]

            # 输出 K 条候选
            for prefix_tokens, orders_step, _ in beams:
                text = self.tokenizer.decode(prefix_tokens) if len(prefix_tokens) > 0 else ""
                results.append((text, orders_step))

        return results

        

    def beam_min_error(self, ta: str, cands, attr: str, N: int):
        """
        评分函数（独立，便于后续替换策略）：
        输入：标准答案文本 ta、beam 候选列表、属性、要检查的前N个候选数
        输出：(best_pred_text, min_error, best_idx)
        逻辑：
        - 将标准答案与前 N 个候选逐一匹配，取 self.metric_FPI 的最小值作为最终 error。
        """
        best_pred, min_err, best_idx = "", float("inf"), -1
        # 只检查前N个候选，同时避免索引越界
        check_count = min(N, len(cands))
        for i in range(check_count):
            pred_txt, _score, _tok_ids = cands[i]
            err = self.metric_FPI(pred_txt, ta, attr)
            if err < min_err:
                min_err, best_pred, best_idx = err, pred_txt, i
        return best_pred, min_err, best_idx


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
            "address_postcode": 7,
            "social_insurance_number": 10,
            "blood_type": 2
        }

        # Record the attribute-wise scores.
        errors = {key: 0 for key in keys}         
        count = {key: 0 for key in keys}
        results = [] # Collect model output
        self.cost.reset()
        self.cost.start()
        
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
            

            for q, ta, attr in zip(questions, true_answers, attributes):
                ol = [attr_lens[attr]]
                self.cost.add_sample(ol[0])
                ans = extract_answer(ta, attr)
                ans = ' ' + ans + '.'

                if self.recover_type == "hf_greedy":
                    mo, od = self.recover_by_hf_greedy([q], output_length=ol, attr_type=attr, answer=ans)[0]
                    err = self.metric_FPI(mo, ta, attr)

                    results.append({
                        "attribute": attr,
                        "question": q,
                        "true_answer": ta,
                        "model_output": mo,
                        "model_order": "",
                        "mode": "hf_greedy",
                        "flip": None,
                        "error": err
                    })
                    errors[attr] += err
                    count[attr] += 1

                elif self.recover_type in ("greedy", "manual_greedy"):
                    mo, od = self.recover_by_greedy([q], output_length=ol, attr_type=attr, answer=ans)[0]
                    err = self.metric_FPI(mo, ta, attr)

                    results.append({
                        "attribute": attr,
                        "question": q,
                        "true_answer": ta,
                        "model_output": mo,
                        "model_order": "",
                        "mode": "manual_greedy",
                        "flip": None,
                        "error": err
                    })
                    errors[attr] += err
                    count[attr] += 1

                elif self.recover_type == "flip":
                    mo, od = self.recover_by_flip([q], output_length=ol, attr_type=attr, answer=ans)[0]
                    err = self.metric_FPI(mo, ta, attr)

                    results.append({
                        "attribute": attr,
                        "question": q,
                        "true_answer": ta,
                        "model_output": mo,
                        "model_order": ", ".join(od),
                        "mode": "flip",
                        "flip": self.flip,
                        "error": err
                    })
                    errors[attr] += err
                    count[attr] += 1

                elif self.recover_type == "beam":
                    cands = self.recover_by_beam([q], output_length=ol, attr_type=attr, answer=ans, K=self.K, C=self.C)
                    
                    Ns = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000]
                    error_list = {}

                    total_cands = len(cands)
                    for N_try in Ns:
                        N_eff = min(N_try, total_cands)
                        best_predN, errN, best_idxN = self.beam_min_error(
                            ta=ta, cands=cands, attr=attr, N=N_eff
                        )
                        error_list[N_try] = {
                            "best_pred": best_predN,
                            "error": errN,
                            "best_idx": int(best_idxN) if best_idxN is not None else None
                        }
                    
                    N_main = min(self.N, total_cands)
                    best_pred, min_err, best_idx = self.beam_min_error(ta=ta, cands=cands, attr=attr, N=N_main)
                    
                    cand_texts = [c[0] for c in cands]
                    cand_scores = [float(c[1]) for c in cands]

                    results.append({
                        "attribute": attr,
                        "question": q,
                        "true_answer": ta,
                        "best_pred": best_pred,
                        "beam_preds": cand_texts,
                        "beam_scores": cand_scores,
                        "mode": "beam",
                        "flip": self.flip,
                        "K": self.K,
                        "C": self.C,
                        "error": min_err,
                        "error_list": error_list
                    })
                    errors[attr] += min_err
                    count[attr] += 1
                
                elif self.recover_type == "grad":
                    mo, od = self.recover_by_grad([q], output_length=ol, attr_type=attr, answer=ans)[0]
                    err = self.metric_FPI(mo, ta, attr)
                    results.append({
                        "attribute": attr, 
                        "question": q, 
                        "true_answer": ta,
                        "model_output": mo, 
                        "model_order": ", ".join(od),
                        "mode": f"grad-{self.recover_mode}",
                        "flip": self.flip, 
                        "error": err
                    })
                    errors[attr] += err
                    count[attr] += 1
                
                else:
                    raise ValueError(f"Unknown recover_type: {self.recover_type}")

        for key in keys:
            errors[key] = errors[key]/count[key] if count[key] else errors[key]
        results.append(errors)
        results.append(count)
        self.cost.stop()
        
        ### Feel free to modify the file name for saving the result.
        # Floder Name
        save_folder = f"{eval_args.unlearnSet}-lr{eval_args.lr_fgt}_WD{eval_args.wd_fgt}_loraRank{eval_args.LoRA_rank_fgt}_loraDrop{eval_args.lora_dropout_fgt}_GradStep{eval_args.grad_acc_steps_fgt}_reg{eval_args.reg_weights_fgt}"
        if eval_args.unlearn_method == "langevin":
            save_folder += f"_noise{eval_args.noise_multiplier_fgt}_clip{eval_args.max_grad_norm_dp_fgt}"
        if eval_args.beta_fgt != 0.1:
            save_folder += f"_beta{eval_args.beta_fgt}"
        if eval_args.modelType in ['unlearned', 'pt-unlearned']:
            save_folder = os.path.join(save_folder, eval_args.unlearn_method)
        # Floder Path
        abs_folder = os.path.join(eval_args.logDIR_recvr, save_folder)
        if not os.path.exists(abs_folder):
            os.makedirs(abs_folder)
        # Json Name
        epoch = eval_args.eps_fgt if eval_args.modelType in ['unlearned', 'pt-unlearned'] else eval_args.epochs
        save_fname = f"recovery-epoch-{epoch}-{eval_args.datasetType}-{self.recover_type}"
        if self.recover_type not in ("hf_greedy", "greedy", "manual_greedy"):
            save_fname += f"{self.flip}"
        if eval_args.quant != "none":
            save_fname += f"_{eval_args.quant}"
        if self.recover_type not in ("flip", "hf_greedy", "greedy", "manual_greedy") and self.K > 1:
            save_fname += f"_K{self.K}_C{self.C}"
            if self.entro:
                save_fname += f"_entro"
        if self.recover_type == "grad":
            save_fname += f"_{self.loss_type}_{self.recover_mode}"
        save_fname += ".json"
        # Json Path
        save_path = os.path.join(abs_folder, save_fname)

        # Save Json
        with open(save_path, "w") as f:
            json.dump(results, f, indent=4, separators=(',', ': '))
        print(f"✅ Saved summary to {save_path}")

        cost_summary = self.cost.summary(
            eval_args=eval_args,
            recover_type=self.recover_type,
            flip=self.flip,
            K=self.K,
            C=self.C,
            N=self.N
        )
        cost_path = save_path.replace(".json", ".cost.json")
        with open(cost_path, "w") as f:
            json.dump(cost_summary, f, indent=4, separators=(',', ': '))
        print(f"✅ Saved cost summary to {cost_path}")


        # Delete Old Style Json
        if self.recover_type == "flip":
            old_fname = f"recovery-epoch-{eval_args.eps_fgt}-{eval_args.datasetType}-flip_logit-{self.flip}.json"
            old_path = os.path.join(abs_folder, old_fname)
            if os.path.exists(old_path):
                try:
                    os.remove(old_path)
                    print(f"🧹 Removed old recovery json: {old_path}")
                except OSError as e:
                    print(f"⚠️ Failed to remove old json {old_path}: {e}")



def main():

    parse = parser_eval()
    parse.add_argument('--flip_logit', type=int, default=None,
                   help='[兼容旧脚本] 等同于 --flip；若提供则覆盖 --flip')
    parse.add_argument('--recover_type', required=True, choices=['hf_greedy', 'manual_greedy', 'greedy', 'flip', 'beam', 'grad'],
                   help="恢复算法类型：hf_greedy、manual_greedy/greedy、flip、beam 或 grad")
    parse.add_argument('--recover_mode', type=str, default='greedy', choices=['greedy', 'oracle'],
                   help="grad 模式：greedy=每步接入预测位；oracle=每步接入真值位")
    parse.add_argument('--flip', type=int, choices=[0, 1], default=1,
                   help="flip 模式下：1 取最大 logit；0 取最小 logit。beam 模式下同理用于每步扩展时的排序方向")
    parse.add_argument('--loss_type', type=str, default='ce', choices=['ce', 'npo'],
                   help="grad 模式下的损失类型：ce 或 npo（仅当 npo 时 beta 生效）")
    parse.add_argument('--beta', type=float, default=1.0,
                   help="grad 模式下的 NPO 损失权重 beta（仅当 loss_type=npo 时生效）")
    parse.add_argument('--K', type=int, default=1, help="beam 模式下保留的序列条数 K")
    parse.add_argument('--C', type=int, default=1, help="beam 模式下每条序列扩展的分支数 C")
    parse.add_argument('--N', type=int, default=1, help="测试时保留的备选集合大小")
    parse.add_argument('--entro', action='store_true', help="是否启用基于熵的自适应权重；默认 False")
    parse.add_argument('--profile_debug', action='store_true', help="在 profiling 中保留 rank/debug 计算；默认关闭以避免污染 decoding timing")
    parse.add_argument('--logDIR_recvr', default="recovery_deepseek_7b_log", type=str)
    eval_args = parse.parse_args()

    # 兼容旧脚本：flip_logit 优先生效
    if getattr(eval_args, 'flip_logit', None) is not None:
        eval_args.recover_type = 'flip'
        eval_args.flip = eval_args.flip_logit

    from evaluate import extract_dir
    modelDIR, dataDIR = extract_dir(eval_args)

    # create folder to save evaluation result
    if not os.path.exists(eval_args.logDIR_recvr):
        os.makedirs(eval_args.logDIR_recvr)
    
    from saved_hf_key import HF_key  # Replace 'HF_key' by your own hugging face key.
    os.environ["HF_TOKEN"] = HF_key

    recover_obj = recoverQA(
        recover_type = eval_args.recover_type,
        recover_mode = eval_args.recover_mode,
        flip = eval_args.flip,
        loss_type = eval_args.loss_type,
        beta = eval_args.beta,
        K = eval_args.K,
        C = eval_args.C,
        N = eval_args.N,
        entro = eval_args.entro,
        profile_debug = eval_args.profile_debug,

        modelDIR = modelDIR,
        eval_batch = 1,
        eval_args = eval_args,
        dataDIR = dataDIR,
        model_name = eval_args.model_name,
        auth_token = HF_key,
    )
    
    recover_obj.evalFPI(eval_args)



if __name__ == "__main__":
    main()
