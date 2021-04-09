import dash
import dash_core_components as dcc
import dash_html_components as html
import core_components as cc
import pandas as pd




#app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options


ddlay_algo = cc.tab_layout('algorithm', 'CNAE', {
        'CNN AutoEncoder':'CNAE',
        'TimeSeries Kmeans': 'TSKM',
        'LSTM AutoEncoder': 'LSAE'
        })
    
def ParameterDiv():
    param_layout = html.Div([
        html.H6("change the value in the text box to see callbacks in action!"),
        html.Div([
            "Algorithms: ",
            ddlay_algo
        ]),
        html.Br(),
        html.Div(id='my-output'),
    ])
    return param_layout



'''
아래 문서를 통해 dash 실습 가능
https://dash.plotly.com/layout
'''
