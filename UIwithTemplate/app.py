# Import required libraries
import pickle
import copy
import pathlib
import urllib.request
import dash
import math
import datetime as dt
import pandas as pd
from dash.dependencies import Input, Output, State, ClientsideFunction
import dash_core_components as dcc
import dash_html_components as html

# Import for algorithm
from params import ParameterDiv
import main_algorithm as MA
import par_clstr_algorithm as pCA
import par_img_data as pid
import par_dtw as pDtw
import core_components as cc
from read_csv import csvDiv, parse_contents
from text_data import textResultDiv
from result_graph import graphDetail, graphCluster, graphBig
from result_graph import GG
import show_detail as sd


app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width"}],
    suppress_callback_exceptions=True
)
server = app.server


# Create app layout
app.layout = html.Div(
    [
        dcc.Store(id="aggregate_data"),
        # empty Div to trigger javascript file for graph resizing
        html.Div(id="output-clientside"),
        html.Div(
            [
                html.Div(
                    [
                        html.Img(
                            src=app.get_asset_url("dash-logo.png"),
                            id="plotly-image",
                            style={
                                "height": "60px",
                                "width": "auto",
                                "margin-bottom": "25px",
                            },
                        )
                    ],
                    className="one-third column",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3(
                                    "Time-series Clustering ",
                                    style={"margin-bottom": "0px"},
                                ),
                                html.H5(
                                    "Kwangwoon Univ. team 일이삼사", style={"margin-top": "0px"}
                                ),
                            ]
                        )
                    ],
                    className="one-half column",
                    id="title",
                ),
                html.Div(
                    [
                        html.Button("학습 시작하기", id="learn-button")
                    ],
                    className="one-third column",
                    id="button",
                ),
            ],
            id="header",
            className="row flex-display",
            style={"margin-bottom": "25px"},
        ),
        # 파라미터 조작, 파일삽입, 군집화 결과 컴포넌트 틀
        html.Div(
            [
                # 파라미터 조작 컴포넌트
                html.Div(
                    [
                        ParameterDiv()
                    ],
                    className="pretty_container four columns",
                    id="cross-filter-options",
                ),
                # 오른쪽 부분 컴포넌트 틀 (파일 삽입, 군집화 결과)
                html.Div(
                    [
                        # 파일 삽입 컴포넌트
                        html.Div(
                            [
                                html.Div(
                                    [csvDiv()],
                                    id="wells",
                                    className="mini_container",
                                )
                            ],
                            id="info-container",
                            className="row container-display",
                        ),
                        # 군집화 결과 그래프 컴포넌트
                        html.Div([
                            html.Div([
                                textResultDiv()
                            ]),
                            html.Div([
                                graphCluster()
                            ], className = 'box-scroll')
                        ],
                            id="countGraphContainer",
                            className="pretty_container",
                        ),
                    ],
                    id="right-column",
                    className="eight columns",
                ),
            ],
            className="row flex-display",
        ),
        # 하단/ 군집화 세부적 결과 그래프 (크게보기, 하나씩 보기)
        html.Div(
            [
                # 세부적 결과 그래프 컴포넌트
                html.Div(
                    id='detail-graph-output'

                ),
                html.Div(
                    sd.detailGraphOption(),
                    className=""
                )
            ],
            className="pretty_container row flex-display",
        ),
    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"},
)


####                                                           ####
# Main Algorithm 에 대한 Layout을 제공해 줍니다. MA: Main Algorithm #
####                                                           ####
@app.callback(
    # my-output id를 가진 컴포넌트의 children 속성으로 들어간다.
    Output(component_id='parma-for-main-algorithm', component_property='children'),
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

####                                                           ####
# 컨트롤 컴포넌트에 의해 세부적 그래프 컴포넌트가 달라집니다. #
####                                                           ####
@app.callback(
    # my-output id를 가진 컴포넌트의 children 속성으로 들어간다.
    Output(component_id='detail-graph-output', component_property='children'),
    Output(component_id='detail-graph-output', component_property='className'),
    Output(component_id='num-of-graphs', component_property='max'),
    Output(component_id='num-of-graphs', component_property='value'),
    Output(component_id='label-n-graphs', component_property='children'),
    # my-input id 를 가진 컴포넌트의 value 속성을 가져온다.
    # Input('detail-graph-submit', 'n_clicks'),
    Input(component_id='nth-cluster', component_property='value'),
    Input(component_id='detail-graph-input', component_property='value'),
    Input(component_id='num-of-graphs', component_property='value')
)
def update_parameter( nth_cluster, detail_graph, num_graph):
    layout = []
    clsName = ''
    nMaxGraphs = len(GG[nth_cluster])
    if num_graph is None or num_graph > nMaxGraphs:
        num_graph = nMaxGraphs
    if detail_graph == 'GrDt':
        layout = graphDetail(nth_cluster, num_graph)
        clsName = "box-scroll"
    elif detail_graph == 'GrBg':
        layout = graphBig(nth_cluster, num_graph)
        clsName = "fullgraph_class"
    #최대 그래프 개수

    return layout, clsName, nMaxGraphs, num_graph, f"Number of data graphs per clusters (max: {nMaxGraphs})"

# Store에 data를 담는다.
@app.callback(
    Output("store-main-algorithm", "data"),
    Input("algorithm", "value"),
)
def store_parameter(ma):
    df = pd.DataFrame()
    df['main_algorithm'] = [ma]
    data = df.to_dict('records')
    return data

# CNN 관련 parameter
@app.callback(
    Output('store-cnn-param', 'data'),
    Input("batch-size", "value"),
    Input("learning-rate", "value"),
    Input("cluster-algorithm", "value"),
    Input("img-data-type", "value"),
)
def store_cnn_param(bs, lr, clag, imgdt):
    df = pd.DataFrame()
    df['batch_size'] = [bs]
    df['learning_rate'] = [lr]
    df['cluster_algorithm'] = [clag]
    df['img-data-type'] = [imgdt]
    data = df.to_dict('records')
    return data

# KMeans 관련 Parameter
@app.callback(
    Output('store-kmeans-param', 'data'),
    Input("number-of-cluster", "value"),
    Input("tolerance", "value"),
    Input("try-n-init", "value"),
    Input("try-n-kmeans", "value"),
    Input("random-center", "value"),
)
def store_kmeans_param(ncl, tol, tni, tnk, rc):
    df = pd.DataFrame()
    df['number_of_cluster'] = [ncl]
    df['tolerance'] = [tol]
    df['try_n_init'] = [tni]
    df['try_n_kmeans'] = [tnk]
    df['random_center'] = [rc]
    data = df.to_dict('records')
    return data

# Image Data(RP) 관련 Parameter
@app.callback(
    Output('store-rp-param', 'data'),
    Input("dimension", "value"),
    Input("time-delay", "value"),
    Input("threshold", "value"),
    Input("percentage", "value"),
)
def store_rp_param(dim, td, th, prtg):
    df = pd.DataFrame()
    df['dimension'] = [dim]
    df['time_delay'] = [td]
    df['threshold'] = [th]
    df['percentage'] = [prtg]
    data = df.to_dict('records')
    return data

# Time Series Kmeans 관련 parameter
@app.callback(
    Output('store-distance-algorithm', 'data'),
    Input('distance-algorithm', "value"),
)
def store_cnn_param(dsag):
    df = pd.DataFrame()
    df['distance-alogrithm'] = [dsag]
    data = df.to_dict('records')
    return data

# CNN 관련 알고리즘 실행!
@app.callback(
    Output("hidden-cnn-div", "children"),
    Input("learn-button", "n_clicks"),
    State("store-main-algorithm", "data"),
    State("store-cnn-param", 'data'),
    State("store-kmeans-param", 'data'),
    State("store-rp-param", 'data'),
    prevent_initial_call=True
)
def get_store_data(n_clicks, ma_data, cnn_data, km_data, rp_data):
    print(ma_data)
    print(cnn_data)
    print(km_data)
    print(rp_data)
    # RP -> CNN -> KMenas알고리즘 적용
    return []

@app.callback(
    Output("hidden-tsk-div", "children"),
    Input("learn-button", "n_clicks"),
    State("store-main-algorithm", "data"),
    State("store-distance-algorithm", 'data'),
    prevent_initial_call=True
)
def get_store_data(n_clicks, ma_data, dis_data):
    print(ma_data)
    print(dis_data)
    return []
# 학습 버튼을 클릭 하게 되면, i
# Main
if __name__ == "__main__":
    app.run_server(debug=True, threaded=True)
