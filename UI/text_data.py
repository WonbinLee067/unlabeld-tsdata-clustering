import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

sample_num = "2";

app.layout = html.Div(children=[
    html.H1(children='군집 개수 : '+sample_num),
    html.H1(children='군집별 데이터 개수 : '+sample_num),
    html.H1(children='총 소요 시간 : '+sample_num),
    html.H1(children='사용된 알고리즘 : '+sample_num),
    html.Div(children='''
        Dash: A web application framework for Python.
    '''),
])

if __name__ == '__main__':
    app.run_server(debug=True)