import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options




def dropdown_layout(id, options, init_value):
    return dcc.Dropdown(
        options = options,
        value=init_value,
        id=id )
def multi_dropdown_layout(id, options, init_value):
    return dcc.Dropdown(
        options = options,
        value = init_value,
        id = id,
        multi = True )

def radio_layout(id, options, init_value):
    return dcc.RadioItems(
        options=options,
        value=init_value,
        id = id )

def checkbox_layout(id, options, init_value):
    return dcc.Checklist(
        options=options,
        value=init_value,
        id = id )

def txt_input_layout(id, init_value='', input_type='text', placeholder='input value'):
    return  dcc.Input(value=init_value, type=input_type, id=id, placeholder=placeholder) # 더 알아봐서 추가

def num_input_layout(id, min=0, max=50, init_value=1, placeholder=""):
    return dcc.Input(value=init_value, id=id, type='number', min=min, max=max)

def slider_layout(id, min, max, marks,init_value=1):
    return dcc.Slider(
        min=min, 
        max=max, 
        marks=marks ,
        value=init_value, 
        id=id)

ddlay_algo = dropdown_layout(
    options=[
        {'label': 'CNN AutoEncoder', 'value': 'CNAE'},
        {'label': 'TimeSeriesKMeans', 'value': 'TSKM'},
        {'label': 'LSTM AutoEncoder', 'value': 'LSAE'}
    ],
    init_value='CNAE',
    id = 'cluster-algorithm')
app.layout = html.Div([
    html.H6("change the value in the text box to see callbacks in action!"),
    html.Div([
        "Algorithms: ",
        ddlay_algo
    ]),
    html.Br(),
    html.Div(id='my-output'),
])

@app.callback(
    # my-output id를 가진 컴포넌트의 children 속성으로 들어간다.
    Output(component_id='my-output', component_property='children'),
    # my-input id 를 가진 컴포넌트의 value 속성을 가져온다.
    Input(component_id='cluster-algorithm', component_property='value')
)
def update_parameter(input_value):
    layout = []
    if input_value == 'CNAE':
        layout = html.Div([
            html.Label('Batch Size'),
            radio_layout('batch-size', [
                {'label': '32', 'value':32},
                {'label': '64', 'value':64}
            ], 32),

            html.Label('learning rate'),
            slider_layout('learning-rate', 1, 3, marks={i: '{}'.format(i) if i == 1 else str(i) for i in range(1, 4)}),

            html.Label('Image Data Type'),
            dropdown_layout('img-data-type', 
                [
                    {'label': 'Recurrence Plot', 'value':'RP'},
                    {'label': 'Raw Data', 'value' : 'RAW'}
                ], 'RP' ),
            
            html.Div(id='param-for-img-data')
        ])
    elif input_value == 'TSKM':
        layout = html.Div([

            html.Label('Cluster 개수'),
            num_input_layout('number-of-cluster', min=2, init_value=2),

            html.Label('거리계산 알고리즘'),
            dropdown_layout('distance-alogrithm', [
                {'label':'Eucleadean', 'value':'EUC'},
                {'label':'DTW', 'value':'DTW'},
                {'label':'Soft-DTW', 'value':'SDT'}
            ], 'DTW'),
            

        ], style={'columnCount': 1})
    elif input_value == 'LSAE':
        layout = html.Div([

        html.Label('Multi-Select Dropdown'),
        html.Label('random_state'),
        dropdown_layout('random-state', [{'label': 'True', 'value':'T'}, {'label': 'False', 'value':'F'}], 'F'),
        ])

    return layout

@app.callback(
    Output(component_id='param-for-img-data', component_property='children'),
    Input(component_id='img-data-type', component_property='value')
)
def image_data_param(data_type):
    params = []
    if data_type == 'RP':
        params = html.Div([
            html.H4("RP Parameters"),
            html.H6("Dimension"),
            html.Label("RP 궤적의 차원수를 결정한다. 공간 궤적 좌표 생성에 쓰이는 데이터 개수이다."),
            num_input_layout('dimension', init_value=1, min=1, placeholder="dimension"),

            html.H6("Time-Delay"),
            html.Label("공간 궤적 좌표 생성시 사용되는 기존 좌표 데이터의 시간 차이를 뜻한다. 따라서 1dim 데이터 사용시 큰 의미가 없다."),
            num_input_layout('time-delay', init_value=1, min=1, placeholder="dimension"),

            html.H6("Threshold"),
            html.Label("궤적의 거리 최솟값을 설정한다."),
            dropdown_layout('threshold',
            [
                {'label': 'float', 'value':'F'},
                {'label': 'point', 'value':'P'},
                {'label': 'distance', 'value':'D'},
                {'label': 'None', 'value' : 'N'}
            ], 'F'), 
            html.Label("percentage if point or distance"),
            slider_layout('percentage', min=10, max=60, marks={i: '{}'.format(i) for i in range(10, 61, 10)})
        ])
    elif data_type == 'RAW': 
        params = html.Div([
            html.H4("RAW Data Preprocessing")
        ])
    return params

if __name__ == '__main__':
    app.run_server(debug=True)


'''
아래 문서를 통해 dash 실습 가능
https://dash.plotly.com/layout
'''
