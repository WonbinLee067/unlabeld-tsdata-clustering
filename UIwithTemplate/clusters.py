import dash_html_components as html

from algorithms import *

def ts_sample_kmeans():
    return html.Div(id='ts-sample-kmeans', children=[
        html.Div(id='hidden-ts-sample-kmeans', style={'display':'none'}),
        kmeans_layout()
    ])
def ts_sample_hierarchy():
    return html.Div(id='ts-sample-hierarchy', children=[
        html.Div(id='hidden-ts-sample-hierarchy', style={'display':'none'}),
        hierarchy_layout()
    ])
def rp_ae_kmeans():
    return html.Div(id='rp-ae-kmeans', children=[
        html.Div(id='hidden-rp-ae-kmeans', style={'display':'none'}),
        rp_layout(),
        autoencoder_layout(),
        kmeans_layout(),
    ])
def rp_ae_hierarchy():
    return html.Div(id='rp-ae-hierarchy', children=[
        html.Div(id='hidden-rp-ae-hierarchy', style={'display':'none'}),
        rp_layout(),
        autoencoder_layout(),
        hierarchy_layout()
    ])
def rp_ae_dbscan():
    return html.Div(id='rp-ae-dbscan', children=[
        html.Div(id='hidden-rp-ae-dbscan', style={'display':'none'}),
        rp_layout(),
        autoencoder_layout(),
        dbscan_layout()
    ])