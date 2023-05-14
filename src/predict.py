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
        probs = [round(pred*100,2), round((1-pred)*100,2)]
        labels = ['toxic', 'non-toxic']
        color_map = {'toxic':"red", "non-toxic":"green"}
        fig = px.bar(x=probs, 
                     y=labels,
                     width=400, height=250, 
                     template="plotly_dark", 
                     text_auto=True, 
                     title="Probabilities(%)",
                     color = labels,
                     color_discrete_map = color_map,
                     labels={'x':'Confidence', 'y':"label"})
        fig.update_traces(width=0.5,textfont_size=20, textangle=0, textposition="outside")
        fig.update_layout(yaxis_title=None,xaxis_title=None, showlegend=False)
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    
    def data_validation(self, data):
        status=True
        for column in data.columns:
            if column not in ['id', 'comment_text']:
                status=False
        return status
                
    def batch_predict(self, data):
        try:
            df = pd.read_csv(data)
            if self.data_validation(df):
                with st.spinner('Please Wait!!! Prediction in process....'):
                    st.success(f'Data Validation Successfull :thumbsup:')
                    df.dropna(inplace=True)
                    df["comment_text"] = df.comment_text.apply(lambda x: re.sub('\n',' ',x).strip())
                    input = self.tokenizer_fn(df.comment_text.values.tolist())
                    preds = self.model.predict(input)
                    df['probabilities'] = preds
                    df['toxic'] = np.where(df['probabilities']>0.5, 1, 0)
                st.success("Prediction Process Completed!!!, :thumbsup:")
                st.info("Press download button to download prediction file")
                return df
            else:
                st.error("Data Validation Failed!! :thumbsdown:")
        except Exception as e:
            print(e)

    def single_predict(self, text:str):
        try:
            text = re.sub('\n',' ',text).strip().lower()
            
            input = self.tokenizer_fn(text)
            pred = self.model.predict(input)[0][0]
            return pred
        except Exception as e:
            print(e)