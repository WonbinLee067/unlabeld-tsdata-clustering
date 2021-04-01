import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State

import main_algorithm as MA
import core_components as cc
from result_graph import *
from result_graph import num_clusters
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


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
                    cc.dropdown_layout("nth-cluster",
                    [
                        {'label': str(i+1), 'value': i}
                        for i in range(num_clusters)
                    ], 0),
                    html.Label("Choose Type of detailed graphs"),
                    cc.dropdown_layout("detail-graph-input", 
                    [
                        {'label': 'Graph Detail', 'value': 'GrDt'},
                        {'label': 'Graph Big', 'value': 'GrBg'}
                    ], 'GrDt'),
                    html.Label("Number of data graphs per clusters"),
                    cc.radio_layout('num-of-graphs', 
                    [
                        {'label': '6', 'value': 6},
                        {'label': '9', 'value': 9},
                        {'label': '12', 'value': 12}
                    ], 6),
                    html.Button('적용하기', id='detail-graph-submit', n_clicks=0)
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
    # my-input id 를 가진 컴포넌트의 value 속성을 가져온다.
    Input('detail-graph-submit', 'n_clicks'),
    State(component_id='nth-cluster', component_property='value'),
    State(component_id='detail-graph-input', component_property='value'),
    State(component_id='num-of-graphs', component_property='value')
)
def update_parameter(n_clicks, nth_cluster, detail_graph, num_graph):
    layout = []
    clsName = ''
    if detail_graph == 'GrDt':
        layout = graphDetail(nth_cluster)
        clsName = "box-scroll"
    elif detail_graph == 'GrBg':
        layout = graphBig(nth_cluster)
        clsName = "fullgraph_class"
    return layout, clsName



if __name__ == '__main__':
    app.run_server(debug=True)