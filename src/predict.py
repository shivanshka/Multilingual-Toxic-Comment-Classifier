import pandas as pd
import numpy as np
import tensorflow as tf
import plotly.express as px
import transformers
from transformers import AutoTokenizer
import os
from src.constants import *
import re
import streamlit as st


class PredictionServices:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def tokenizer_fn(self, text:str):
        tokens = self.tokenizer(text, 
                                max_length=MAX_LEN, 
                                truncation=True, 
                                padding="max_length",
                                add_special_tokens=True,
                                return_tensors="tf",
                                return_token_type_ids = False)
        inputs = dict(tokens)
        return inputs
    
    def plot(self, pred):
        fig = px.bar(x=[round(pred), round(1-pred)], 
                     y=['toxic', 'non-toxic'],
                     width=500, height=250, 
                     template="plotly_dark", 
                     text_auto='1', 
                     title="Probabilities(%)")
        fig.update_traces(width=0.3,textfont_size=15, textangle=0, textposition="outside")
        fig.update_layout(yaxis_title=None,xaxis_title=None)
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    
    def data_validation(data):
        df = pd.read_csv(data)
        status=True
        for column in df.columns:
            if column not in ['id', 'comment_text']:
                status=False
        return status
                
    def batch_predict(self, data):
        try:
            df = pd.read_csv(data)
            df.dropna(inplace=True)
            df = df.comment_text.apply(lambda x: re.sub('\n',' ',x).strip())
            input = self.tokenizer_fn(df.comment_text.values.tolist())
            preds = self.model.predict(input)
            df['probabilities'] = preds
            df['toxic'] = np.where(df['probabilities']>0.5, 1, 0)
            return df
        except Exception as e:
            print(e)

    def single_predict(self, text:str):
        try:
            text = re.sub('\n',' ',text).strip()
            input = self.tokenizer_fn(text)
            pred = self.model.predict(input)[0][0]
            return pred
        except Exception as e:
            print(e)