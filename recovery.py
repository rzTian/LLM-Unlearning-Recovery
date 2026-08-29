import json
import torch
import math
from transformers import LogitsProcessorList
import os
import re

from argsetting import parser_eval
from evaluate import EvalQA, build_unlearn_child_folder
from utils import CustomizedLogitsProcessor, get_hf_token

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
        self.recover_type = recover_type.lower().strip()
        self.recover_mode = recover_mode if recover_mode in ('greedy', 'oracle') else 'greedy'
        self.flip = int(flip)
        self.loss_type = loss_type.lower().strip()
        self.beta = float(beta)
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
        Compute closed-form last-layer gradient norms for candidate tokens.
        """
        import torch
        device = logits_last.device
        p = torch.softmax(logits_last, dim=-1).squeeze(0)  # [V]
        p_sq_sum = torch.sum(p * p)
        h = hidden_last.squeeze(0).squeeze(0)              # [H]
        h_norm = torch.norm(h, p=2) + 1e-12

        use_npo = (self.loss_type == 'npo')
        if use_npo:
            if ref_logits_last is None:
                self.ref_model.eval()
                ref_out = self.ref_model(
                    input_ids=self._last_input_ids,
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
            ce_vec_norm = torch.sqrt(1.0 - 2.0 * p_t + p_sq_sum + 1e-12)
            scale = 1.0
            if use_npo:
                q_t = q[tid].clamp_min(eps)
                num = 2.0 * torch.pow(p_t, beta)
                den = torch.pow(p_t, beta) + torch.pow(q_t, beta)
                scale = (num / den).detach()
            g_norm = (scale * ce_vec_norm * h_norm).item()
            results.append((tid, g_norm))
        return results

    def _should_stop_attr(self, attr_type, out_tok_ids, new_tok_id):
        if new_tok_id == 13:
            return True

        text = self.tokenizer.decode(out_tok_ids, skip_special_tokens=False)

        if attr_type == "year_of_birth":
            return re.fullmatch(r"\s*\d{4}\.", text) is not None

        elif attr_type == "address_postcode":
            return re.fullmatch(r"\s*[A-Z]\d[A-Z]\d[A-Z]\d\.", text) is not None

        elif attr_type == "social_insurance_number":
            digits = re.sub(r"\D", "", text)
            return len(digits) >= 9 or text.endswith(".")

        elif attr_type == "blood_type":
            return text.strip() in {"A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"} \
                or text.strip().endswith(".")
        
        return False

    def get_token_rank(self, logits, token_id, selected_token_ids, step=None):
        """
        Compute and print the rank of token_id among selected_token_ids, highlighting:
        A = answer token, T = top-1 logit token.
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
                markers += "A"
            if idx == top1_idx:
                markers += "T"

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
            orders_all = []
            generated_history = []
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
                processor.history = generated_history[:step]

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
                current_token = outputs_step.sequences[:, -1:].item()
                generated_history.append(current_token)

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

        # Concatenate all the output IDs
        full_output_ids = torch.cat(outputs, dim=1)  # shape: (batch, total_steps)
        decoded_outputs = self.tokenizer.batch_decode(full_output_ids, skip_special_tokens=False)
        print("token_ids:", full_output_ids)
        print("decoded:", decoded_outputs)

        results = list(zip(decoded_outputs, orders_all))
        return results


    def recover_by_beam(self, questions, output_length, attr_type, answer, K=5, C=3):
        """
        Beam search over attribute-constrained candidate tokens.
        """
        assert attr_type is not None, "attr_type must be provided."
        assert questions and isinstance(questions, (list, tuple)), "questions must be a non-empty list or tuple."
        max_steps = output_length[0] if output_length else 20

        def _print_topk(step: int, beams, tag: str = "beam"):
            K_now = len(beams)
            if K_now == 0:
                print(f"[{tag}] step {step+1}: (empty)")
                return

            scores = [b[0] for b in beams]
            max_score = max(scores)
            rel = [math.exp(s - max_score) for s in scores]
            denom = sum(rel)
            rel = [r / max(denom, 1e-45) for r in rel]

            print(f"[{tag}] step {step+1} Top-{K_now}:")
            for idx, (score, _in_ids, _attn, out_tok_ids) in enumerate(beams, start=1):
                seq_len = len(out_tok_ids)
                avg_score = (score / max(seq_len, 1)) if seq_len > 0 else float('nan')
                text = self.tokenizer.decode(out_tok_ids, skip_special_tokens=False)
                print(
                    f"    [{idx}] score={score:.3f}  avg_score={avg_score:.3f}  "
                    f"rel_p={rel[idx-1]:.3e}  text='{text}'"
                )

        inputs = self.tokenizer(
            questions,
            padding=True,
            truncation=True,
            max_length=150,
            return_tensors="pt",
        ).to(self.device)

        beams = [(
            0.0,
            inputs["input_ids"].clone(),
            inputs["attention_mask"].clone(),
            []
        )]

        with torch.no_grad():
            finished_beams = []

            for step in range(max_steps):
                new_beams = []

                for log_score, input_ids, attention_mask, out_tok_ids in beams:
                    processor = CustomizedLogitsProcessor(
                        tokenizer=self.tokenizer,
                        attr_type=attr_type,
                        generation_step=step,
                        flip_logit=self.flip,
                    )
                    processor.history = out_tok_ids
                    lp_list = LogitsProcessorList([processor])

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True
                    )
                    next_logits = outputs.logits[:, -1, :]

                    next_logits = lp_list(input_ids, next_logits)[0]

                    sel_ids = getattr(processor, "selected_token_ids", None)
                    if not sel_ids:
                        sel_ids = torch.arange(next_logits.size(-1), device=next_logits.device, dtype=torch.long)
                    else:
                        sel_ids = torch.as_tensor(list(sel_ids), device=next_logits.device, dtype=torch.long)

                    sel_logits = next_logits[sel_ids]
                    print(f"[debug] step={step} sel_logits min={sel_logits.min().item():.2f} max={sel_logits.max().item():.2f}")
                    scores = sel_logits
                    log_probs = torch.log_softmax(sel_logits, dim=-1)
                    probs = torch.exp(log_probs)

                    alpha = 1.0
                    if self.entro:
                        H = -(probs * log_probs).sum()
                        H_max = math.log(probs.numel()) if probs.numel() > 0 else 1.0
                        alpha = 1.0 - float(H / H_max) if H_max > 0 else 1.0
                        alpha = max(0.1, min(1.0, alpha))
                        print(f'H:{H}, Hmax:{H_max}, alpha:{alpha}')

                    topC = min(C, scores.numel())
                    _, top_idx = torch.topk(scores, k=topC, largest=True, sorted=True)

                    eps = 1e-12
                    for j in range(topC):
                        idx = top_idx[j]
                        tok_id = sel_ids[idx].item()
                        incr = float(scores[idx])
                        new_log_score = log_score + alpha * incr
                        new_out_tok_ids = out_tok_ids + [tok_id]

                        tok = torch.tensor([[tok_id]], device=input_ids.device, dtype=input_ids.dtype)
                        new_input_ids = torch.cat([input_ids, tok], dim=-1)
                        new_attention_mask = torch.cat([attention_mask, torch.ones_like(tok)], dim=-1)

                        if self._should_stop_attr(attr_type, new_out_tok_ids, tok_id):
                            finished_beams.append((new_log_score, input_ids, attention_mask, new_out_tok_ids))
                        else:
                            new_beams.append((new_log_score, input_ids, attention_mask, new_out_tok_ids))

                if not new_beams:
                    break

                new_beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:K]
                beams = new_beams

                _print_topk(step, beams, tag=f"beam-{self.flip}")
            
            if finished_beams:
                beams = sorted(finished_beams, key=lambda x: x[0], reverse=True)[:K]

        results = []
        for log_score, _input_ids, _attn, out_tok_ids in beams:
            decoded = self.tokenizer.decode(out_tok_ids, skip_special_tokens=True)
            results.append((decoded, log_score, out_tok_ids))

        results.sort(key=lambda x: x[1], reverse=True)
        return results
    

    def recover_by_grad(self, questions, output_length, attr_type, answer):
        """
        Recover tokens by ranking candidate last-layer gradient norms.
        """
        device = self.model.device
        self._ensure_pad_token()

        if self.loss_type == 'npo' and self.ref_model is None:
            print("[WARN] loss_type='npo' requires ref_model; falling back to 'ce'.")
            self.loss_type = 'ce'

        L = int(output_length[0]) if isinstance(output_length, (list, tuple)) else int(output_length)
        L = max(L, 1)

        answer_ids_full = self.tokenizer(answer, add_special_tokens=False)["input_ids"]
        if len(answer_ids_full) == 0:
            print(f"[WARN] empty answer '{answer}'; oracle mode cannot teacher-force.")
        if len(answer_ids_full) < L:
            L = len(answer_ids_full) if len(answer_ids_full) > 0 else L

        results = []

        for q in questions:
            enc = self.tokenizer(q, return_tensors="pt", add_special_tokens=False)
            base_ids = enc["input_ids"].to(device)
            base_attn = torch.ones_like(base_ids, device=device)

            K = int(getattr(self, "K", 1))
            C = int(getattr(self, "C", 1))
            beams = [([], [], 0.0)]

            for step in range(L):
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
                        print(f"[WARN] no candidates for {key}; falling back to digits.")
                    cand_ids = self._digit_token_ids()
                else:
                    cand_ids = sorted(set(int(x) for x in selected))

                true_id = int(answer_ids_full[step]) if step < len(answer_ids_full) else None
                choose_max = (self.flip == 1)
                new_beams = []

                for prefix_tokens, orders_step, acc_score in beams:
                    if len(prefix_tokens) > 0:
                        input_ids, attn = self._concat_inputs(base_ids, prefix_tokens)
                    else:
                        input_ids, attn = base_ids, base_attn

                    self.model.eval()
                    out = self.model(
                        input_ids=input_ids,
                        attention_mask=attn,
                        output_hidden_states=True,
                        use_cache=False
                    )
                    logits_last = out.logits[:, -1, :].float()             # [1, V]
                    hidden_last = out.hidden_states[-1][:, -1, :].float()  # [1, H]

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

                    norms = self._grad_norms_last_layer(logits_last, hidden_last, cand_ids, ref_logits_last)

                    pairs = sorted(norms, key=lambda x: x[1], reverse=choose_max)
                    direction = "max" if choose_max else "min"
                    lt = self.loss_type.upper()
                    print(f"\n[Grad-{self.recover_mode}|{lt}|beam<={K}, C={C}] step={step+1}/{L} {direction}")
                    for tid, g in pairs:
                        mark = []
                        if true_id is not None and tid == true_id: mark.append("<true>")
                        toks = self._safe_decode(tid)
                        print(f"{repr(toks)}:{tid:>6}  -> {g:.6f} {' '.join(mark)}")

                    take = min(C, len(pairs))
                    top_c = pairs[:take]
                    for tid, g in top_c:
                        new_prefix = list(prefix_tokens)
                        if self.recover_mode == 'greedy':
                            new_prefix.append(int(tid))
                        else:  # oracle
                            if true_id is not None:
                                new_prefix.append(int(true_id))

                        rank_list = [tid_ for tid_, _ in pairs]
                        if true_id in rank_list:
                            orders_str = f"{rank_list.index(true_id)}/{len(rank_list)-1}"
                        else:
                            orders_str = f"NA/{len(rank_list)-1}"
                        new_orders = list(orders_step) + [orders_str]

                        new_score = acc_score + (g if choose_max else -g)

                        new_beams.append((new_prefix, new_orders, new_score))

                new_beams.sort(key=lambda x: x[2], reverse=True)
                beams = new_beams[:K]

            for prefix_tokens, orders_step, _ in beams:
                text = self.tokenizer.decode(prefix_tokens) if len(prefix_tokens) > 0 else ""
                results.append((text, orders_step))

        return results

        

    def beam_min_error(self, ta: str, cands, attr: str, N: int):
        """
        Return the lowest-error candidate among the first N beam outputs.
        """
        best_pred, min_err, best_idx = "", float("inf"), -1
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

        errors = {key: 0 for key in keys}         
        count = {key: 0 for key in keys}
        results = []
        
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
                ans = ' ' + ans + '.'

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
        
        save_folder = build_unlearn_child_folder(eval_args)
        if eval_args.modelType in ['unlearned', 'pt-unlearned']:
            save_folder = os.path.join(save_folder, eval_args.unlearn_method)
        abs_folder = os.path.join(eval_args.logDIR_recvr, save_folder)
        if not os.path.exists(abs_folder):
            os.makedirs(abs_folder)

        epoch = eval_args.eps_fgt if eval_args.modelType in ['unlearned', 'pt-unlearned'] else eval_args.epochs
        save_fname = f"recovery-epoch-{epoch}-{eval_args.datasetType}-{self.recover_type}-flip{self.flip}"
        if getattr(eval_args, "quant", "none") != "none":
            save_fname += f"_{eval_args.quant}"
        if self.recover_type == "beam":
            save_fname += f"_K{self.K}_C{self.C}"
            if self.entro:
                save_fname += "_entro"
        if self.recover_type == "grad":
            save_fname += f"_{self.loss_type}_{self.recover_mode}"
        save_fname += ".json"
        save_path = os.path.join(abs_folder, save_fname)

        with open(save_path, "w") as f:
            json.dump(results, f, indent=4, separators=(',', ': '))
        print(f"Saved summary to {save_path}")


        if self.recover_type == "flip":
            old_fname = f"recovery-epoch-{eval_args.eps_fgt}-{eval_args.datasetType}-flip_logit-{self.flip}.json"
            old_path = os.path.join(abs_folder, old_fname)
            if os.path.exists(old_path):
                try:
                    os.remove(old_path)
                    print(f"Removed old recovery json: {old_path}")
                except OSError as e:
                    print(f"Failed to remove old json {old_path}: {e}")



def main():

    parse = parser_eval()
    parse.add_argument('--flip_logit', type=int, default=None,
                   help='Backward-compatible alias for --flip; overrides --flip when set.')
    parse.add_argument('--recover_type', required=True, choices=['flip', 'beam', 'grad'],
                   help="Recovery method: flip, beam, or grad.")
    parse.add_argument('--recover_mode', type=str, default='greedy', choices=['greedy', 'oracle'],
                   help="Grad mode: greedy uses predicted prefixes; oracle uses gold prefixes.")
    parse.add_argument('--flip', type=int, choices=[0, 1], default=1,
                   help="For flip/beam: 1 selects the largest transformed logit, 0 selects the smallest.")
    parse.add_argument('--loss_type', type=str, default='ce', choices=['ce', 'npo'],
                   help="Grad loss type: ce or npo.")
    parse.add_argument('--beta', type=float, default=1.0,
                   help="NPO beta for grad recovery.")
    parse.add_argument('--K', type=int, default=1, help="Beam width.")
    parse.add_argument('--C', type=int, default=1, help="Branches kept per beam step.")
    parse.add_argument('--N', type=int, default=1, help="Number of candidates to score.")
    parse.add_argument('--entro', action='store_true', help="Enable entropy-based adaptive weighting.")
    parse.add_argument('--logDIR_recvr', default="recovery_deepseek_7b_log", type=str)
    eval_args = parse.parse_args()

    # Preserve old scripts that pass --flip_logit.
    if getattr(eval_args, 'flip_logit', None) is not None:
        eval_args.recover_type = 'flip'
        eval_args.flip = eval_args.flip_logit

    from evaluate import extract_dir
    modelDIR, dataDIR = extract_dir(eval_args)

    if not os.path.exists(eval_args.logDIR_recvr):
        os.makedirs(eval_args.logDIR_recvr)
    
    HF_key = get_hf_token()
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
