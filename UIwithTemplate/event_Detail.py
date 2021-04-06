import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State

import main_algorithm as MA
import core_components as cc
from result_graph import *
from result_graph import GG
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

num_clusters = len(GG)

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

app.layout = html.Div([
     html.Div([
          # 군집화 결과 그래프 컴포넌트
        html.Div(
            html.Div(
                graphCluster(),
                className = 'box-scroll'
            ),
            id="countGraphContainer",
            className="pretty_container",
        ),
        html.Div([
            # 세부적 결과 그래프 컴포넌트
            html.Div(
                #보기1
                # [graphDetail()]
                # className="pretty_container seven columns",
                #보기2
                # [graphBig()],
                # className="pretty_container seven columns fullgraph_class",
                id='detail-graph-output'
            ),
            # 세부적 결과 그래프 선택 조작 컴포넌트
            ## 여기서 컴포넌트를 조작하여 위 세부적 결과 그래프 형태를 선택한다.
            html.Div(
                [
                    html.Label("Choose cluster"),
                    cc.radio_layout("nth-cluster", 
                    [
                        {'label': str(i+1), 'value': i}
                        for i in range(num_clusters)
                    ], 0),
                    # cc.dropdown_layout("nth-cluster",
                    # [
                    #     {'label': str(i+1), 'value': i}
                    #     for i in range(num_clusters)
                    # ], 0),
                    html.Label("Choose Type of detailed graphs"),
                    cc.dropdown_layout("detail-graph-input", 
                    [
                        {'label': 'Graph Detail', 'value': 'GrDt'},
                        {'label': 'Graph Big', 'value': 'GrBg'}
                    ], 'GrDt'),
                    html.Label("Number of data graphs per clusters", id="label-n-graphs"),
                    cc.num_input_layout('num-of-graphs', min=1, init_value=1),
                    # cc.radio_layout('num-of-graphs', 
                    # [
                    #     {'label': '6', 'value': 6},
                    #     {'label': '9', 'value': 9},
                    #     {'label': '12', 'value': 12}
                    # ], 6),
                    # html.Button('적용하기', id='detail-graph-submit', n_clicks=0)
                ],
                className="",
            )], 
            className="pretty_container row flex-display", 
        )
    ])
])


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
    if num_graph > nMaxGraphs:
        num_graph = nMaxGraphs
    if detail_graph == 'GrDt':
        layout = graphDetail(nth_cluster, num_graph)
        clsName = "box-scroll"
    elif detail_graph == 'GrBg':
        layout = graphBig(nth_cluster, num_graph)
        clsName = "fullgraph_class"
    #최대 그래프 개수
    
    return layout, clsName, nMaxGraphs, num_graph, f"Number of data graphs per clusters (max: {nMaxGraphs})"
# input 으로 클러스터 넘버를 얻어옴
# output : Max 값을 알려줌
# output: Max 값이 넘었을때, 값을 고쳐줌
# output  : 


if __name__ == '__main__':
    app.run_server(debug=True)