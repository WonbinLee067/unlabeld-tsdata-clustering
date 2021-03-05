import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# graph: dictionary 형태의 데이터
# n: 한 군집 내 시계열 데이터 개수
# label: 딕셔너리 keys(list형태)
# color: 그래프 line 색
def makeGraph(graph, n, label, color):
    df=[]
    for i in range(0, n):
        df.append(graph[label[i]])

    fig = go.Figure()
    for i in range(0, n):
        fig.add_trace(go.Scatter(y=df[i], name=label[i], line=dict(color=color)))

    return fig
    
# figure: makeGraph()를 이용해 만든 그래프
# label: 그래프 이름
def updateLayout(figure, name, yaxis='value'):
    figure.update_layout(
        title=name,
        yaxis_title=yaxis
    )


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


GG = {
    'gr1':[0,1,2,5,4,8],
    'gr2':[5,2,4,6,5,4,3,7],
    'gr3':[4,11,2,6,5,1,2,7],
    'gr4':[7,4,5,7,4,1]
}

colors = {
    'background': 'dimgray',
    'text': 'white'
}

fig = makeGraph(GG, 4, list(GG.keys()), 'teal')
updateLayout(fig, 'GG1')

def graphDiv():
    graph = html.Div(style={'backgroundColor': colors['background']}, children=[
        html.H1(
            children='AXISX',
            style={
                'textAlign': 'center',
                'color': colors['text']
            }
        ),

        dcc.Graph(
            id='GG',
            figure=fig
        )
    ])
    return graph


