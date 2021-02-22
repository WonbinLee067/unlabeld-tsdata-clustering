import dash_core_components as dcc
import dash_html_components as html
import core_components as cc

def param_kmeans():
    params = html.Div([
            html.Label('Cluster 개수'),
            cc.num_input_layout('number-of-cluster', min=2, init_value=2)
        ])
    return params

def param_dbscan():
    params = html.Div()

    return params