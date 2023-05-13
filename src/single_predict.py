import pandas as pd
import numpy as np
import tensorflow as tf
import transformers
from transformers import AutoTokenizer
import os
from src.constants import *
import re


class SinglePrediction:
    def __init__(self):
        self.model = tf.keras.models.load_model(MODEL_PATH)
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    def tokenizer(self, text:str):
        tokens = self.tokenizer(text, 
                                max_length=MAX_LEN, 
                                truncation=True, 
                                padding="max_length",
                                add_special_tokens=True,
                                return_tensors="tf",
                                return_token_type_ids = False)
        return dict(tokens)

    def predict(self, text:str):
        try:
            text = re.sub('\n',' ',text).strip()
            input = self.tokenizer(text)
            preds = self.model.predict(input)[0][0]
            return preds
        except Exception as e:
            print(e)