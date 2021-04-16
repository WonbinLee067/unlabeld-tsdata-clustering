import dash_core_components as dcc
import dash_html_components as html
import core_components as cc

def param_kmeans():
    params = html.Div([
            dcc.Store(id='store-kmeans-param', data=[]),
            html.Label('Cluster 개수'),
            cc.num_input_layout('number-of-cluster', min=2, init_value=2),
            html.Label('Tolerance, default = 1e-4'),
            cc.radio_layout('tolerance', [
                {'label': '1e-4', 'value': 'O_F'},
            ], 'O_F'),
            html.Label('KMeans를 시도해볼 횟수'),
            cc.num_input_layout('try-n-init', min=1, max=50, init_value=10),
            html.Label('Kmeans가 알고리즘 안에 반복되는 최대 횟수'),
            cc.num_input_layout('try-n-kmeans', min=10, max=1000, init_value=300),
            html.Label('중심 랜덤으로 지정하기'),
            cc.switch_layout('random-center', label="랜덤 사용"),
        ])
    return params

def param_dbscan():
    params = html.Div()

    return params
