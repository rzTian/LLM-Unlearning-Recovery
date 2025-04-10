import json
import torch
import os
from transformers import LogitsProcessor
import re


class CustomizedLogitsProcessor(LogitsProcessor):

    def __init__(self, tokenizer, tokenType, flip_logit):
        
        self.tokenizer = tokenizer
        self.tokenType = tokenType
        self.selected_token_ids = self._get_token_ids()
        self.flip_logit = flip_logit
       
    def _get_token_ids(self):

        if self.tokenType == "digits":
            
            tokenized_string = self.tokenizer("0123456789.")
            selected_tokens = tokenized_string['input_ids'] 
            
            # remove special token ids
            # special_token_ids = [29871, 29889]  # ["_", "."]

            special_token_ids = [1, 29871, 29889]  # ["<s>", "_", "."]            
            selected_tokens = [t for t in selected_tokens if t not in special_token_ids]          

        elif self.tokenType == "blood_type":

            unique_token_ids = set()  # Use a set to avoid duplicates        
            strings = ["A+.", "A-.", "B+.", "B-.", "AB+.", "AB-.", "O+.", "O-."]

            for string in strings:
                tokens = self.tokenizer.tokenize(string)
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                unique_token_ids.update(token_ids)  # Add new token IDs to the set
            
            selected_tokens = list(unique_token_ids)
        
        else:
            raise ValueError
        
        '''[Self note] 
        Saving the tokens is for sanity check purpose, may be removed later ...
        '''
        folder_name = f"tokens_for_{self.tokenType}"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        with open(os.path.join(folder_name, "llama_tokens.json"), "w") as f:
            json.dump(selected_tokens, f, indent=4)
        
        return selected_tokens


    def __call__(self, input_ids, scores):

        mask = torch.full_like(scores, fill_value=-1e10)

        for i in range(scores.size(0)):  
            # Copy over only selected token ids
            mask[i, self.selected_token_ids] = -scores[i, self.selected_token_ids] if self.flip_logit else scores[i, self.selected_token_ids]
        return mask
    
