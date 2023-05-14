import streamlit as st
import pandas as pd
import plotly.express as px
from src import *
import os

global model
global prediction

@st.cache_resource
def model_obj():
    model = ModelLoader()
    prediction = PredictionServices(model.Model, model.Tokenizer)
    st.image(os.path.join("img","toxic.jpg"))
    return prediction
    
prediction = model_obj()

def single_predict(text):
    preds = prediction.single_predict(text)
    if preds < 0.5:
        st.success(f'Non Toxic Comment!!! :thumbsup:')
    else:
        st.error(f'Toxic Comment!!! :thumbsdown:')
    prediction.plot(preds)

def batch_predict(data):
    preds = prediction.batch_predict(data)
    return preds.to_csv(index=False).encode('utf-8')
    
st.title('Toxic Comment Classifier')
st.write("This application will help to classify any comment or text in any language into 'TOXIC' or 'NON-TOXIC'")
tab1, tab2 = st.tabs(["Single Value Prediciton","Batch Prediction"])

with tab1:
    st.subheader("Prediction")
    with st.form("comment_form", clear_on_submit=True):
        comment = st.text_area(label="Enter your comment")
        button = st.form_submit_button(label="Predict")
        if button:
            with st.spinner('Please Wait!!! Prediction in process....'):
                single_predict(comment)

with tab2:
    st.subheader("Batch Prediction")
    csv_file = st.file_uploader("Upload File",type=['csv'])

    if csv_file is not None:
        csv = batch_predict(csv_file)
        st.download_button(
            label="Download",
            data=csv,
            file_name='prediction.csv',
            mime='text/csv',
        )