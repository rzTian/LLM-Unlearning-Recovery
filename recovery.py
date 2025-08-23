import json
import torch
import math
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, LogitsProcessorList
from peft import PeftModel, LoraConfig, get_peft_model
import torch.nn.functional as F
from typing import List, Tuple
import os
import logging
import argparse
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


class recoverQA(EvalQA):
    
    def __init__(self, recover_type, recover_mode='greedy', flip=1, loss_type='ce', beta=1.0,
                 K=1, C=1, N=1, entro=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recover_type = recover_type.lower().strip()  # 统一小写
        self.recover_mode = recover_mode if recover_mode in ('greedy', 'oracle') else 'greedy'  # 目前仅用于 grad-recover
        self.flip = int(flip)  # 1=最大logit，0=最小logit
        self.loss_type = loss_type.lower().strip()  # 'ce' 或 'npo'
        self.beta = float(beta)  # NPO 权重参数
        self.K = int(K)
        self.C = int(C)
        self.N = int(N)
        self.entro = entro

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

    def recover_by_flip(self, questions, output_length, attr_type, answer):
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
                    flip_logit=self.flip
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

        # ========== 小工具：每步打印 Top-K ==========
        def _print_topk(step:int, beams, tag:str="beam"):
            """
            打印当前步的 Top-K：
            - logp: 累积log概率（越大越好）
            - avg_nll: 平均负对数似然
            - ppl: 困惑度（越小越好）
            - rel_p: 在本步的相对概率（log-sum-exp归一化），不会下溢
            """
            K_now = len(beams)
            if K_now == 0:
                print(f"[{tag}] step {step+1}: (empty)")
                return

            # 收集分数与文本
            logps = [b[0] for b in beams]  # (log_score, input_ids, attn, out_ids)
            max_logp = max(logps)
            # log-sum-exp 归一化为相对概率（本步可比性强）
            rel = [math.exp(lp - max_logp) for lp in logps]
            denom = sum(rel)
            rel = [r / max(denom, 1e-45) for r in rel]

            print(f"[{tag}] step {step+1} Top-{K_now}:")
            for idx, (logp, _in_ids, _attn, out_tok_ids) in enumerate(beams, start=1):
                seq_len = len(out_tok_ids)
                avg_nll = (-logp / max(seq_len, 1)) if seq_len > 0 else float('inf')
                ppl = math.exp(avg_nll) if seq_len > 0 else float('inf')
                text = self.tokenizer.decode(out_tok_ids, skip_special_tokens=False)
                # 用科学计数法避免显示为0
                print(f"    [{idx}] logp={logp:.3f}  avg_nll={avg_nll:.3f}  ppl={ppl:.3f}  rel_p={rel[idx-1]:.3e}  text='{text}'")

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
        with torch.no_grad():
            for step in range(max_steps):
                candidates = []

                for log_score, input_ids, attention_mask, out_tok_ids in beams:
                    # 构造 processor（带 flip 与属性约束）
                    processor = CustomizedLogitsProcessor(
                        tokenizer=self.tokenizer,
                        attr_type=attr_type,
                        generation_step=step,
                        flip_logit=self.flip,  # 0/1
                    )
                    lp_list = LogitsProcessorList([processor])

                    # 前向计算 logits（取最后一位）
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True
                    )
                    next_logits = outputs.logits[:, -1, :]  # shape: [1, vocab]

                    # 应用 processor（HF 约定：传入 input_ids 与 logits）
                    next_logits = lp_list(input_ids, next_logits)[0]  # -> shape: [vocab]

                    # 只考虑 processor 指定的候选集合；若 processor 未给出，则用全词表
                    sel_ids = getattr(processor, "selected_token_ids", None)
                    if not sel_ids:
                        sel_ids = torch.arange(next_logits.size(-1), device=next_logits.device, dtype=torch.long)
                    else:
                        # 关键：统一成 LongTensor（避免后续索引返回 python int）
                        sel_ids = torch.as_tensor(list(sel_ids), device=next_logits.device, dtype=torch.long)

                    sel_logits = next_logits[sel_ids]  # [num_selected]
                    print(f"[debug] step={step} sel_logits min={sel_logits.min().item():.2f} max={sel_logits.max().item():.2f}")
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
                    top_scores, top_idx = torch.topk(scores, k=topC, largest=True, sorted=True)

                    eps = 1e-12
                    for j in range(topC):
                        # 现在 sel_ids[...] 是 0-dim Tensor，可以安全 .item()
                        idx = top_idx[j]
                        tok_id = sel_ids[idx].item()
                        incr = float(log_probs[idx])
                        new_log_score = log_score + alpha * incr

                        tok = torch.tensor([[tok_id]], device=input_ids.device, dtype=input_ids.dtype)
                        new_input_ids = torch.cat([input_ids, tok], dim=-1)
                        new_attention_mask = torch.cat([attention_mask, torch.ones_like(tok)], dim=-1)

                        candidates.append((
                            new_log_score,
                            new_input_ids,
                            new_attention_mask,
                            out_tok_ids + [int(tok_id)],
                        ))

                # 若没有候选（过度约束），提前结束
                if not candidates:
                    break

                # 全局排序，保留 K
                candidates.sort(key=lambda x: x[0], reverse=True)
                beams = candidates[:min(K, len(candidates))]

                # 每一位扩展完成后打印当前 Top-K
                _print_topk(step, beams, tag=f"beam-{self.flip}")

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
            

            for q, ta, attr in zip(questions, true_answers, attributes):
                ol = [attr_lens[attr]]
                ans = extract_answer(ta, attr)

                if self.recover_type == "flip":
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
                    best_pred, min_err, best_idx = self.beam_min_error(ta=ta, cands=cands, attr=attr, N=self.N)
                    if attr == 'blood_type': min_err = best_idx
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
                        "error": min_err
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
        
        ### Feel free to modify the file name for saving the result.
        # Floder Name
        save_folder = f"{eval_args.unlearnSet}-lr{eval_args.lr_fgt}_WD{eval_args.wd_fgt}_loraRank{eval_args.LoRA_rank_fgt}_loraDrop{eval_args.lora_dropout_fgt}_reg{eval_args.reg_weights_fgt}"
        if eval_args.modelType == 'unlearned':
            save_folder = os.path.join(save_folder, eval_args.unlearn_method)
        # Floder Path
        abs_folder = os.path.join(eval_args.logDIR, save_folder)
        if not os.path.exists(abs_folder):
            os.makedirs(abs_folder)
        # Json Name
        save_fname =  f"recovery-epoch-{eval_args.eps_fgt}-{eval_args.datasetType}-{self.recover_type}{self.flip}"
        if self.recover_type != "flip" and self.K > 1:
            save_fname += f"_K{self.K}_C{self.C}"
            if self.entro:
                save_fname += f"_entro"
        if self.recover_type == "grad":
            save_fname += f"_{self.recover_mode}"
        save_fname += ".json"
        # Json Path
        save_path = os.path.join(abs_folder, save_fname)

        # Save Json
        with open(save_path, "w") as f:
            json.dump(results, f, indent=4, separators=(',', ': '))
        print(f"✅ Saved summary to {save_path}")


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
    parse.add_argument('--recover_type', required=True, choices=['flip', 'beam', 'grad'],
                   help="恢复算法类型：flip 或 beam 或 grad")
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
    eval_args = parse.parse_args()

    # 兼容旧脚本：flip_logit 优先生效
    if getattr(eval_args, 'flip_logit', None) is not None:
        eval_args.recover_type = 'flip'
        eval_args.flip = eval_args.flip_logit

    from evaluate import extract_dir
    modelDIR, dataDIR = extract_dir(eval_args)

    eval_args.logDIR = "recovery_llama_7b_log"
    # create folder to save evaluation result
    if not os.path.exists(eval_args.logDIR):
        os.makedirs(eval_args.logDIR)
    
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