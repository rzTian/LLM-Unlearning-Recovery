import os
from transformers import AutoTokenizer

class Dummy:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _build_token_sets(self):
        sets = {}
        encode = lambda s: self.tokenizer(s, add_special_tokens=False)["input_ids"]

        sets["digits"] = list(set(sum([encode(c) for c in "0123456789"], [])))
        sets["upper"] = list(set(sum([encode(c) for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"], [])))
        sets["1or2"] = list(set(sum([encode(c) for c in ["1", "2"]], [])))
        sets["9or0"] = list(set(sum([encode(c) for c in ["9", "0"]], [])))
        sets["ABO"] = list(set(sum([encode(bt) for bt in ["A", "B", "O", "AB"]], [])))
        sets["blood_type"] = list(set(sum([encode(bt) for bt in [
            "A+.", "A-.", "B+.", "B-.", "O+.", "O-.", "AB+.", "AB-."
        ]], [])))
        
        special_token_ids = [1, 29871, 29889]  # ["<s>", "_", "."]
        for key in sets:
            sets[key] = [tid for tid in sets[key] if tid not in special_token_ids]

        os.makedirs("tokens_notes", exist_ok=True)
        with open("tokens_notes/all_token_sets.txt", "w") as f:
            for name, ids in sets.items():
                f.write(f"=== {name} ===\n")
                for token_id in ids:
                    token_str = self.tokenizer.decode([token_id])
                    f.write(f"{repr(token_str)} : {token_id}\n")
                f.write("\n")

        return sets

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

        # Maximum token positions per attribute (determined empirically or by format)
        attr_lens = {
            "year_of_birth": 5,
            "address_postcode": 7,
            "social_insurance_number": 10,
            "blood_type": 2
        }

        os.makedirs("tokens_notes", exist_ok=True)
        note_path = "tokens_notes/attr_token_sets.txt"

        with open(note_path, "w") as fout:
            for filepath in glob.glob(f"{attr_dir}/*.jsonl"):
                attr_name = os.path.splitext(os.path.basename(filepath))[0]
                max_len = attr_lens.get(attr_name, 6)

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

    def print_token_info(self, sentence):
        """
        对输入句子进行 tokenize，逐个打印 token、token_id 及其位置
        """
        tokens = self.tokenizer.tokenize(sentence, add_special_tokens=False)
        token_ids = self.tokenizer(sentence, add_special_tokens=False)["input_ids"]

        print(f"\nInput sentence:\n{sentence}\n")
        print("Token breakdown:")
        for i, (tok, tid) in enumerate(zip(tokens, token_ids)):
            print(f"Token {i:02d}: {repr(tok):<15} ID: {tid}")


        
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-chat")  # 替换为你的模型
    dummy = Dummy(tokenizer)

    token_sets = dummy._build_attr_token_sets()
    print("✅ Token sets saved to tokens_notes/all_token_sets.txt")

    # sentence = "[INST] What is the postcode of Tom Lopez's address? [/INST] Tom Lopez's address postcode is T9N5K2. Lopez's address postcode is T9N5K2.L5."
    # dummy.print_token_info(sentence)
