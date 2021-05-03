import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq


def kmeans_layout():
    return html.Div(id='kmeans-param', children=[
        dcc.Store(id='store-kmeans-param', data=[]),
        html.H5("KMeans Parameters"),
        html.Label('Cluster 개수'),
        dcc.Input(id='number-of-cluster', min=2, value=2, type='number'),
        html.Label('Tolerance, default = 1e-4'),
        dcc.RadioItems(id='tolerance', 
            options=[
                {'label': '1e-4', 'value': 'O_F'},
            ], value='1e-4'),
        html.Label('KMeans를 시도해볼 횟수'),
        dcc.Input(id='try-n-init', min=1, value=10, type='number'),
        html.Label('Kmeans가 알고리즘 안에 반복되는 최대 횟수'),
        dcc.Input(id='try-n-kmeans', min=10, value=300, type='number'),
        html.Label('중심 랜덤으로 지정하기'),
        daq.BooleanSwitch(id='random-center', on=False, label="랜덤 사용", labelPosition='top'),
        html.Hr()
    ])
def dbscan_layout():
    return html.Div(id='dbscan-param', children=[
        dcc.Store(id='store-dbscan-param', data=[]),
        html.H5("DBSCAN Parameters"),
        html.Label('Epsilon 크기'),
        dcc.Input(id='dbscan-epsilon', min=0, max=1, value=0.5, type='number'),
        html.Label('min-sample 크기(정수)'),
        dcc.Input(id='dbscan-min-sample', min=1, value=5, type='number'),
        html.Hr()
    ])
def hierarchy_layout():
    return html.Div(id='hierarchy-param', children=[
        dcc.Store(id='store-hierarchy-param', data=[]),
        html.H5("Hierarchy Parameters"),
        html.Label('Cluster 개수'),
        dcc.Input(id='number-of-cluster', min=2, value=2, type='number'),
        html.Label('n-init'),
        dcc.Input(id='try-n-init', min=1, value=10, type='number'),
        html.Label('linkage'),
        dcc.Dropdown(
            id='linkage',
            options=[
                {'label': 'ward', 'value': 'ward'}
            ],
            value='ward'),
        html.Hr()
    ])
def rp_layout():
    return html.Div(id='rp-param', children=[
        dcc.Store(id='store-rp-param', data=[]),
        html.H5("RP Parameters"),
        html.H6("Dimension"),
        html.Label("RP 궤적의 차원수를 결정한다. 공간 궤적 좌표 생성에 쓰이는 데이터 개수이다."),
        dcc.Input(id='dimension', value=1, min=1, type='number'),

        html.H6("Time-Delay"),
        html.Label("공간 궤적 좌표 생성시 사용되는 기존 좌표 데이터의 시간 차이를 뜻한다. 따라서 1dim 데이터 사용시 큰 의미가 없다."),
        dcc.Input(id='time-delay', value=1, min=1, type='number'),

        html.H6("Threshold"),
        html.Label("궤적의 거리 최솟값을 설정한다."),
        dcc.Dropdown(id='threshold',
        options=[
            {'label': 'float', 'value':'F'},
            {'label': 'point', 'value':'P'},
            {'label': 'distance', 'value':'D'},
            {'label': 'None', 'value' : 'N'}
        ], value='F'),
        html.Label("percentage if point or distance"),
        dcc.Slider(id='percentage', min=10, max=60, marks={i: '{}'.format(i) for i in range(10, 61, 10)}, value=1, step=1),
        html.Hr()
    ])
def autoencoder_layout():
    return html.Div(id='autoencoder-param', children=[
        dcc.Store(id='store-autoencoder-param', data=[]),
        html.H5("Autoencoder Parameters"),
        html.Label('Batch Size'),
        dcc.RadioItems(id='autoencoder-batch-size', 
            options=[
                {'label': '32', 'value':32},
                {'label': '64', 'value':64}
            ], value=32),
        html.Label('learning rate'),
        dcc.Slider(id='autoencoder-learning-rate', min=1, max=3, marks={i: '{}'.format(i) if i == 1 else str(i) for i in range(1, 4)}),
        html.Label('loss function'),
        dcc.Dropdown(id='autoencoder-loss-function',
            options=[
                {'label': 'rmse', 'value':'RMSE'}
            ], value='RMSE'),
        html.Label('activation function'),
        dcc.Dropdown(id='autoencoder-activation-function',
            options=[
                {'label': 'sigmoid', 'value':'sigmoid'}
            ], value='sigmoid'),
        html.Hr()
    ])