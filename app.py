import gradio as gr
from src import *

model = ModelLoader()
prediction = PredictionServices(model.Model, model.Tokenizer)

def single_predict(text):
    print(text)
    preds = prediction.single_predict(text)
    toxic_pred = float(preds)
    non_toxic_pred = float(1-toxic_pred)
    rslt = {"Toxic":toxic_pred,"Non Toxic":non_toxic_pred}
    return rslt

app = gr.Interface(inputs=gr.Textbox(label="Enter Comment"), 
                   fn=single_predict, 
                   outputs=[gr.Label('Probabilities')],
                   title="Toxic Comment Classifier")

app.launch()


