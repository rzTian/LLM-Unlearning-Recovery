import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
import os


class data_preprocess:
    def __init__(self, model_name, auth_token = None, special_format = True):
        self.model_name = model_name
        self.auth_token = auth_token       
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.auth_token)        
        # Llama model does not provide pad_token. Using eos_token as the pad token instead.
        self.tokenizer.pad_token = self.tokenizer.eos_token 
         # Special chat format for Llama model
        self.Question_startToken = "[INST] " if special_format else ""
        self.Question_endToken = " [/INST]" if special_format else ""


    def load_dataset(self, dataDIR):
        with open(dataDIR, "r") as file:
            qa_data = json.load(file)
        return qa_data

    def format_QA(self, question, answer):
        format_question = self.Question_startToken + question + self.Question_endToken
        return format_question + " " +  answer

    # Tokenize dataset
    def tokenize_function(self, examples):
        formatted_texts = [self.format_QA(q,a) for q,a in zip(examples["question"], examples["answer"])]
        inputs = self.tokenizer(formatted_texts, padding="max_length", truncation=True, max_length=200)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]  # 1 = real token, 0 = padding

        labels = []
        for q, ids, attn_mask in zip(examples["question"], input_ids, attention_mask):
            # Tokenize the question separately to get its length
            question_text = self.Question_startToken + q + self.Question_endToken
            question_ids = self.tokenizer(question_text, add_special_tokens=False)["input_ids"]

            Q_end = len(question_ids)  # Determine the lenght of the tokenized question

            # Mask question tokens (-100) and keep answer tokens
            label_seq = [-100] * Q_end + ids[Q_end:]


            # Ensure **ONLY** padding tokens are ignored, NOT <EOS> tokens
            label_seq = [l if attn == 1 else -100 for l, attn in zip(label_seq, attn_mask)]

            labels.append(label_seq)

        inputs["labels"] = labels
        return inputs
    

    def tokenize_datasetQA(self, qa_data):
    
        questions = [item["question"] for item in qa_data]
        answers = [item["answer"] for item in qa_data]
        dataset =  Dataset.from_dict({"question": questions, "answer": answers})
        tokenized_datasets = dataset.map(self.tokenize_function, batched=True, remove_columns=["question", "answer"])

        return tokenized_datasets

