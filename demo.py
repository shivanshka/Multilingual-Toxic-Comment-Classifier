
import pandas as pd
import plotly.express as px


probs = [round(0.978*100,2), round((1-0.978)*100,2)]
labels = ['toxic', 'non-toxic']
color_map = {'toxic':"red", "non-toxic":"green"}
fig = px.bar(x=probs, 
                y=labels,
                width=600, height=250, 
                template="plotly_dark", 
                text_auto='1', 
                title="Probabilities(%)",
                color = labels,
                color_discrete_map = color_map,
                labels={'x':'Confidence', 'y':"label"})

fig.update_traces(width=0.3,textfont_size=15, textangle=0, textposition="outside")
fig.update_layout(yaxis_title=None,xaxis_title=None, )
fig.show()
