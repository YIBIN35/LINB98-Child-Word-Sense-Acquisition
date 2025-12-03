# embeddings.py
import torch
import numpy as np
from transformers import BertTokenizerFast, BertModel

class BertEmbedder:
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name, output_hidden_states=True)
        self.model.eval()
        self.mask_token = "[MASK]"


    def create_vectors_unmasked(self, tokens):
        inputs = self.tokenizer(tokens, is_split_into_words=True,
                                return_tensors="pt", truncation=True, max_length=512)

        hidden_vectors = self.model(
            **inputs, output_hidden_states=True
        ).hidden_states
        vec = torch.stack(hidden_vectors[-6:]).mean(dim=0).squeeze(0)
        vec = vec[1:-1]
        return vec.mean(dim=0).detach().numpy()

    def create_vectors_masked(self, sentence, target_word):
        tokens = sentence.split()
        tokens = [self.mask_token if t == target_word else t for t in tokens]
        return self.create_vectors_unmasked(tokens)
