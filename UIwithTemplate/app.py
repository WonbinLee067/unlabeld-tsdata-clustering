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
import show_detail as sd


# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()

app = dash.Dash(
    __name__, 
    meta_tags=[{"name": "viewport", "content": "width=device-width"}],
    suppress_callback_exceptions=True
)
server = app.server


# Download pickle file
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/plotly/datasets/master/dash-sample-apps/dash-oil-and-gas/data/points.pkl",
    DATA_PATH.joinpath("points.pkl"),
)
points = pickle.load(open(DATA_PATH.joinpath("points.pkl"), "rb"))


# Load data
df = pd.read_csv(
    "https://github.com/plotly/datasets/raw/master/dash-sample-apps/dash-oil-and-gas/data/wellspublic.csv",
    low_memory=False,
)
df["Date_Well_Completed"] = pd.to_datetime(df["Date_Well_Completed"])
df = df[df["Date_Well_Completed"] > dt.datetime(1960, 1, 1)]

trim = df[["API_WellNo", "Well_Type", "Well_Name"]]
trim.index = trim["API_WellNo"]
dataset = trim.to_dict(orient="index")


# Create global chart template
mapbox_access_token = "pk.eyJ1IjoicGxvdGx5bWFwYm94IiwiYSI6ImNrOWJqb2F4djBnMjEzbG50amg0dnJieG4ifQ.Zme1-Uzoi75IaFbieBDl3A"

layout = dict(
    autosize=True,
    automargin=True,
    margin=dict(l=30, r=30, b=20, t=40),
    hovermode="closest",
    plot_bgcolor="#F9F9F9",
    paper_bgcolor="#F9F9F9",
    legend=dict(font=dict(size=10), orientation="h"),
    title="Satellite Overview",
    mapbox=dict(
        accesstoken=mapbox_access_token,
        style="light",
        center=dict(lon=-78.05, lat=42.54),
        zoom=7,
    ),
)

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
                        html.A(
                            html.Button("Learn More", id="learn-more-button"),
                            href="https://plot.ly/dash/pricing/",
                        )
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
                # # 세부적 결과 그래프 선택 조작 컴포넌트
                # ## 여기서 컴포넌트를 조작하여 위 세부적 결과 그래프 형태를 선택한다.
                # html.Div(
                #     sd.detailGraphOption(),
                #     className="pretty_container five columns"
                # ),
            ],
            className="pretty_container row flex-display",
        ),
        # html.Div(
        #     [
        #         html.Div(
        #             [dcc.Graph(id="pie_graph")],
        #             className="pretty_container seven columns",
        #         ),
        #         html.Div(
        #             [dcc.Graph(id="aggregate_graph")],
        #             className="pretty_container five columns",
        #         ),
        #     ],
        #     className="row flex-display",
        # ),
    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"},
)


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

####                                                           ####
# 컨트롤 컴포넌트에 의해 세부적 그래프 컴포넌트가 달라집니다. #
####                                                           ####
@app.callback(
    # my-output id를 가진 컴포넌트의 children 속성으로 들어간다.
    Output(component_id='detail-graph-output', component_property='children'),
    Output(component_id='detail-graph-output', component_property='className'),
    # my-input id 를 가진 컴포넌트의 value 속성을 가져온다.
    Input('detail-graph-submit', 'n_clicks'),
    State(component_id='detail-graph-input', component_property='value'),
    State(component_id='num-of-graphs', component_property='value')
)
def update_parameter(n_clicks, detail_graph, num_graph):
    layout = []
    clsName = ''
    if detail_graph == 'GrDt':
        layout = graphDetail()

        #clsName = "pretty_container seven columns box-scroll"
    elif detail_graph == 'GrBg':
        layout = graphBig()
        clsName = "fullgraph_class"
    return layout, clsName

# Main
if __name__ == "__main__":
    app.run_server(debug=True)
