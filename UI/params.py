import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd

import core_components as cc
import main_algorithm as MA
import par_clstr_algorithm as pCA
import par_img_data as pid

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options


ddlay_algo = cc.tab_layout('algorithm', 'CNAE', {
        'CNN AutoEncoder':'CNAE',
        'TimeSeriesKmeans': 'TSKM',
        'LSTM AutoEncoder': 'LSAE'
        })
    
app.layout = html.Div([
    html.H6("change the value in the text box to see callbacks in action!"),
    html.Div([
        "Algorithms: ",
        ddlay_algo
    ]),
    html.Br(),
    html.Div(id='my-output'),
])

####                                                           ####
# Main Algorithm 에 대한 Layout을 제공해 줍니다. MA: Main Algorithm #
####                                                           ####
@app.callback(
    # my-output id를 가진 컴포넌트의 children 속성으로 들어간다.
    Output(component_id='my-output', component_property='children'),
    # my-input id 를 가진 컴포넌트의 value 속성을 가져온다.
    Input(component_id='algorithm', component_property='value')
)
def update_parameter(input_value):
    layout = []
    if input_value == 'CNAE':
        layout = MA.cnn_autoencoder()
    elif input_value == 'TSKM':
        layout = MA.time_sereies_kmeans()
    elif input_value == 'LSAE':
        layout = MA.lstm_autoencoder()
    return layout

####                                              ####
# Clustering Algorithm에 대한 Parameter 를 제공합니다. #
# pCA : params for Clustering Algorithm              #
####                                              ####
@app.callback(
    Output('param-for-cluster-algorithm', 'children'),
    Input('cluster-algorithm', 'value')
)
def clister_algorithm_param(cluster):
    params = []
    if cluster == 'KMS':
        params = pCA.param_kmeans()
    elif cluster == 'DBS':
        params = pCA.param_dbscan()
    return params

####                                                           ####
# CNN Auto Encoder의 데이터 형식으로, image data의 형식을 결정합니다.#
# pid : params for image data                                     #
####                                                           ####
@app.callback(
    Output(component_id='param-for-img-data', component_property='children'),
    Input(component_id='img-data-type', component_property='value')
)
def image_data_param(data_type):
    params = []
    if data_type == 'RP':
        params = pid.recurrence_plot()
    elif data_type == 'RAW': 
        params = pid.raw_img()
    return params

if __name__ == '__main__':
    app.run_server(debug=True)


'''
아래 문서를 통해 dash 실습 가능
https://dash.plotly.com/layout
'''
