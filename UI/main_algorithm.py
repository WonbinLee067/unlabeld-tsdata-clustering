import dash_core_components as dcc
import dash_html_components as html
import core_components as cc

def cnn_autoencoder():
    layout = html.Div([
            html.Label('Batch Size'),
            cc.radio_layout('batch-size', [
                {'label': '32', 'value':32},
                {'label': '64', 'value':64}
            ], 32),
            html.Label('learning rate'),
            cc.slider_layout('learning-rate', 1, 3, marks={i: '{}'.format(i) if i == 1 else str(i) for i in range(1, 4)}),

            html.Label('Clustering Algorithm'),
            cc.dropdown_layout('cluster-algorithm', [
                {'label': 'KMeans', 'value': 'KMS'},
                {'label': 'DBSCAN', 'value': 'DBS'}
            ], 'KMS'),
            html.Div(id='param-for-cluster-algorithm'),
            html.Label('Image Data Type'),
            cc.dropdown_layout('img-data-type', 
                [
                    {'label': 'Recurrence Plot', 'value':'RP'},
                    {'label': 'Raw Data', 'value' : 'RAW'}
                ], 'RP' ),
            
            html.Div(id='param-for-img-data')
        ])
    return layout

def time_sereies_kmeans():
    layout = html.Div([

            html.Label('Cluster 개수'),
            cc.num_input_layout('number-of-cluster', min=2, init_value=2),
            html.Label('거리계산 알고리즘'),
            cc.dropdown_layout('distance-alogrithm', [
                {'label':'Eucleadean', 'value':'EUC'},
                {'label':'DTW', 'value':'DTW'},
                {'label':'Soft-DTW', 'value':'SDT'}
            ], 'DTW'),
            html.Div(id='param-for-dtw'),
            html.Label('Tolerance, default = 1e-4'),
            cc.radio_layout('tolerance', [
                {'label': '1e-4', 'value': 'O_F'},
            ], 'O_F'), 
        ], style={'columnCount': 1})
    return layout

def lstm_autoencoder():
    layout = html.Div([

        html.Label('Multi-Select Dropdown'),
        html.Label('random_state'),
        cc.dropdown_layout('random-state', [{'label': 'True', 'value':'T'}, {'label': 'False', 'value':'F'}], 'F'),
    ])
    return layout