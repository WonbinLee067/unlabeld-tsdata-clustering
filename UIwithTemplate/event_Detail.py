import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

from clusters import *
import pandas as pd
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

# 각 세부 알고리즘 dropdown! 
## TimeSereisRandom + KMeans
## TimeSereisRandom + 계층적군집화
## TimeSereisRandom + TimeSereisKmeans
## Rp+autoencoder+kmeans
## Rp+autoencoder+계층군집
## RP+autoencoder+DBSCAN

# 세부 기능 별 parameter
## TimeSereisRandom
#### size
## KMeans
#### cluster 개수
#### n_init (Kmeans 시도할 횟수(함수 내부에서))
#### max_iter (각 kmeans 시도할때 마다 반복할 횟수)
#### tol (수렴 관용점..)
## 계층적 군집화
#### cluster 개수
#### n_init (Kmeans 시도할 횟수(함수 내부에서))
#### linkage ? 
## DBSCAN
#### epsilon
#### min_samples
## RP 알고리즘
#### dimension
#### time_delay
#### threshold
#### percentage
#### flatten
## Autoencoder
#### size
#### optimizer
#### learning_rate
#### loss_function
#### activation_function

app.layout = html.Div([
    dcc.Dropdown(
        id='main-cluster-algorithm',
        options=[
                    {'label': 'TimeSeriesSample + KMeans', 'value':'ts_sample_kmeans'},
                    {'label': 'TimeSeriesSample + Hierarchical Cluster', 'value':'ts_sample_hierarchy'},
                    {'label': 'TimeSeriesSample + TimeSeriesKMeans', 'value':'ts_sample_ts_kmeans'},
                    {'label': 'RP + Autoencoder + Kmeans', 'value':'rp_ae_kmeans'},
                    {'label': 'RP + Autoencoder + Hierarchical Cluster', 'value':'rp_ae_hierarchy'},
                    {'label': 'RP + Autoencoder + DBSCAN', 'value':'rp_ae_dbscan'},
                ],
        value='ts_sample_kmeans'),
    html.Div(id='parameter-layout'),
    html.Button("학습 시작하기", id="learn-button")
])

@app.callback(
    Output('parameter-layout', 'children'),
    Input('main-cluster-algorithm', 'value')
)
def select_main_algorithm(algorithm):
    if algorithm == 'ts_sample_kmeans':
        return ts_sample_kmeans()
    elif algorithm == 'ts_sample_hierarchy':
        return ts_sample_hierarchy()
    elif algorithm == 'rp_ae_kmeans':
        return rp_ae_kmeans()
    elif algorithm == 'rp_ae_hierarchy':
        return rp_ae_hierarchy()
    elif algorithm == 'rp_ae_dbscan':
        return rp_ae_dbscan()

#######################################################################
#  각 알고리즘 별 변수 저장
# KMeans 관련 parameter
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
# hirarchy cluster 관련 parameter
@app.callback(
    Output('store-hierarchy-param', 'data'),
    Input("number-of-cluster", "value"),
    Input("try-n-init", "value"),
    Input("linkage", "value"),
)
def store_hirarchy_param(ncl, tni, lnk):
    df = pd.DataFrame()
    df['number_of_cluster'] = [ncl]
    df['try_n_init'] = [tni]
    df['linkage'] = [lnk]
    data = df.to_dict('records')
    return data
# DBSCAN 관련 parameter
@app.callback(
    Output('store-dbscan-param', 'data'),
    Input("dbscan-epsilon", "value"),
    Input("dbscan-min-sample", "value")
)
def store_dbscan_param(eps, msp):
    df = pd.DataFrame()
    df['epsilon'] = [eps]
    df['min_sample'] = [msp]
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

# Autoencoder (ae) 관련 Parameter
@app.callback(
    Output('store-autoencoder-param', 'data'),
    Input("autoencoder-batch-size", "value"),
    Input("autoencoder-learning-rate", "value"),
    Input("autoencoder-loss-function", "value"),
    Input("autoencoder-activation-function", "value"),
)
def store_ae_param(bs, lr, loss_f, act_f):
    df = pd.DataFrame()
    df['batch_size'] = [bs]
    df['learning_rate'] = [lr]
    df['loss_function'] = [loss_f]
    df['activation_function'] = [act_f]
    data = df.to_dict('records')
    return data
#######################################################################
## 군집화 알고리즘 별 파라미터 호출
# timeSeriesSample + kmeans
@app.callback(
    Output("hidden-ts-sample-kmeans", "children"),
    Input("learn-button", "n_clicks"),
    State("store-kmeans-param", 'data'),
    prevent_initial_call=True 
)
def exct_ts_sample_kmeans(n_clicks, km_data):
    print(km_data)
    return []
# timeSeriesSample + hierarchy
@app.callback(
    Output("hidden-ts-sample-hierarchy", "children"),
    Input("learn-button", "n_clicks"),
    State("store-hierarchy-param", 'data'),
    prevent_initial_call=True 
)
def exct_ts_sample_kmeans(n_clicks, hrc_data):
    print(hrc_data)
    return []
# rp-ae-kmeans
@app.callback(
    Output("hidden-rp-ae-kmeans", "children"),
    Input("learn-button", "n_clicks"),
    State("store-rp-param", 'data'),
    State("store-autoencoder-param", 'data'),
    State("store-kmeans-param", 'data'),
    prevent_initial_call=True 
)
def exct_rp_autoencoder_kmeans(n_clicks, rp_data, ae_data, km_data):
    print(rp_data)
    print(ae_data)
    print(km_data)
    return []
# rp-ae-hierarchy
@app.callback(
    Output("hidden-rp-ae-hierarchy", "children"),
    Input("learn-button", "n_clicks"),
    State("store-rp-param", 'data'),
    State("store-autoencoder-param", 'data'),
    State("store-hierarchy-param", 'data'),
    prevent_initial_call=True 
)
def exct_rp_autoencoder_hierarchy(n_clicks, rp_data, ae_data, hrc_data):
    print(rp_data)
    print(ae_data)
    print(hrc_data)
    return []
# rp-ae-dbscan
@app.callback(
    Output("hidden-rp-ae-dbscan", "children"),
    Input("learn-button", "n_clicks"),
    State("store-rp-param", 'data'),
    State("store-autoencoder-param", 'data'),
    State("store-dbscan-param", 'data'),
    prevent_initial_call=True 
)
def exct_rp_autoencoder_dbscan(n_clicks, rp_data, ae_data, dbs_data):
    print(rp_data)
    print(ae_data)
    print(dbs_data)
    return []


if __name__ == '__main__':
    app.run_server(debug=True)