import dash_html_components as html
import core_components as cc

def detailGraphOption():
    return [
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
    ]