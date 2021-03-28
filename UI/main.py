import base64
import datetime
import io

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State

from params import ParameterDiv
import main_algorithm as MA
import par_clstr_algorithm as pCA
import par_img_data as pid
import par_dtw as pDtw
import core_components as cc
from read_csv import csvDiv, parse_contents
from text_data import textResultDiv
from result_graph import graphDiv
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
app.layout = html.Div([
#title Div
html.Div([
        html.H1("Timeseries Clustering")
    ]),
    #main content
    html.Div([
        #left
        html.Div([
            ParameterDiv()
        ]),
        #right
        html.Div([
            #read_csv
            html.Div([
                csvDiv()
            ]),
            #result_div
            html.Div([
                #text_data div
                html.Div([
                    textResultDiv()
                ]),
                html.Div([
                    graphDiv()
                ])
            ])
        ]),
        
    ])
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

####                                           ####
# DTW / soft-DTW 에 대한 특정 파라미터를 생성합니다. #
####                                           ####
@app.callback(
    Output(component_id='param-for-dtw', component_property='children'),
    Input(component_id='distance-alogrithm', component_property='value')
)
def image_data_param(data_type):
    params = []
    if data_type == 'DTW':
        params = pDtw.dtw()
    elif data_type == 'SDT': 
        params = pDtw.soft_dtw()
    return params

##read_csv
@app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))             
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children    

if __name__ == '__main__':
    app.run_server(debug=True)