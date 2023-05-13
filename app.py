import streamlit as st
import pandas as pd
from src import *

single = SinglePrediction()
batch = BatchPrediction()

def single_predict(text):
    preds, fig = single.predict(text)

    if preds < 0.5:
        st.success(f'Non Toxic Comment!!! :thumbsup:')
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    else:
        st.error(f'Toxic Comment!!! :thumbsup:')
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

def batch_predict(data):
    if batch.data_validation(data):
        st.success(f'Data Validation Successfull :thumbsup:')
        preds = batch.predict(data)
        return preds.to_csv(index=False).encode('utf-8')
    else:
        st.error(f'Data Validation Failed :thumbsdown:')

st.title('Toxic Comment Classifier')
menu = ["Single Value Prediciton","Batch Prediction"]
choice = st.sidebar.radio("Menu",menu)

if choice=="Single Value Prediciton":
    st.subheader("Prediction")
    #comment = st.text_input("Comment", 'Enter your comment here') 
    #trigger = st.button('Predict', on_click=single_predict(comment))
    form = st.form("my_form")
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



