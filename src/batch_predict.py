import pandas as pd
import numpy as np
import tensorflow as tf
import transformers
from transformers import AutoTokenizer
import os
from src.constants import *
import re


class BatchPrediction:
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
    
    def data_validation(data):
        df = pd.read_csv(data)
        status=True
        for column in df.columns:
            if column not in ['id', 'comment_text']:
                status=False
        return status
                
    def predict(self, data):
        try:
            df = pd.read_csv(data)
            df.dropna(inplace=True)
            df = df.comment_text.apply(lambda x: re.sub('\n',' ',x).strip())
            input = self.tokenizer(df.comment_text.values.tolist())
            preds = self.model.predict(input)
            df['probabilities'] = preds
            df['toxic'] = np.where(df['probabilities']>0.5, 1, 0)
            return df
        except Exception as e:
            print(e)