import pandas as pd
import numpy as np
import tensorflow as tf
import plotly.express as px
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
    
    def plot(self, pred):
        fig = px.bar(x=[round(pred), round(1-pred)], 
                     y=['toxic', 'non-toxic'],
                     width=500, height=250, 
                     template="plotly_dark", 
                     text_auto='1', 
                     title="Probabilities(%)")
        fig.update_traces(width=0.3,textfont_size=15, textangle=0, textposition="outside")
        fig.update_layout(yaxis_title=None,xaxis_title=None)
        return fig

    def predict(self, text:str):
        try:
            text = re.sub('\n',' ',text).strip()
            input = self.tokenizer(text)
            pred = self.model.predict(input)[0][0]
            fig = self.plot(pred)
            return pred, fig
        except Exception as e:
            print(e)