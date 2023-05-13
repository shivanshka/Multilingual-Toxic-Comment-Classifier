import gradio as gr
from src import *

model = ModelLoader()
prediction = PredictionServices(model.Model, model.Tokenizer)

def single_predict(text):
    preds = prediction.single_predict(text)
    return {"toxic":preds,"non-toxic":(1-preds)}

app = gr.Interface(gr.Textbox(label="Enter Comment"), 
                   inputs=single_predict, 
                   outputs=gr.Label('probabilities'))

app.launch()


