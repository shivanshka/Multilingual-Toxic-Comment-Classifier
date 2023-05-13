import streamlit as st
import pandas as pd
import plotly.express as px
from src import *

model = ModelLoader()
prediction = PredictionServices(model.Model, model.Tokenizer)

def single_predict(text):
    preds = prediction.single_predict(text)

    if preds < 0.5:
        st.success(f'Non Toxic Comment!!! :thumbsup:')
    else:
        st.error(f'Toxic Comment!!! :thumbsdown:')

    prediction.plot(preds)

def batch_predict(data):
    if prediction.data_validation(data):
        st.success(f'Data Validation Successfull :thumbsup:')
        preds = prediction.batch_predict(data)
        return preds.to_csv(index=False).encode('utf-8')
    else:
        st.error(f'Data Validation Failed :thumbsdown:')

st.title('Toxic Comment Classifier')
menu = ["Single Value Prediciton","Batch Prediction"]
choice = st.sidebar.radio("Menu",menu)

if choice=="Single Value Prediciton":
    st.subheader("Prediction")
    form = st.form("comment_form")
    comment = form.text_input("Enter comment")
    form.form_submit_button("Predict",on_click=single_predict(comment))
else:
    st.subheader("Batch Prediction")
    csv_file = st.file_uploader("Upload Image",type=['csv','parquet'])

    if csv_file is not None:
        csv = batch_predict(csv_file)
        st.download_button(
            label="Predict and Download",
            data=csv,
            file_name='prediction.csv',
            mime='text/csv',
        )



