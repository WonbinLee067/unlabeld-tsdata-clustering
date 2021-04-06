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
        html.Label("Number of data graphs per clusters", id="label-n-graphs"),
        cc.num_input_layout('num-of-graphs', min=1, init_value=1),
    ]