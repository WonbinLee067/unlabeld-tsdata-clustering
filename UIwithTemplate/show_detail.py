import dash_html_components as html
import core_components as cc
from result_graph import num_clusters
def detailGraphOption():
    return [
        html.Label("Choose cluster"),
        cc.radio_layout("nth-cluster", 
        [
            {'label': str(i+1), 'value': i}
            for i in range(num_clusters)
        ], 0),
        html.Label("Choose Type"),
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
    ]