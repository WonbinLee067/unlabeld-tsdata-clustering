import dash_core_components as dcc
import dash_html_components as html
import core_components as cc

def dtw():
    params = html.Div([
        html.Label('path 구하는 알고리즘 돌리는 횟수'),
        cc.num_input_layout('try_n_barycenter'),
    ])
    return params

def soft_dtw():
    params = html.Div([
        html.Label('path 구하는 알고리즘 돌리는 횟수'),
        cc.num_input_layout('try_n_barycenter'),
        html.Label('Metric Gammas 높을 수록 부드러우지지만, 시간이 걸림'),
        cc.slider_layout('metric_gamma', min=0, max=1, step=0.1,
        marks={i/10: '{}'.format(i/10) if i != 0 else '0' for i in range(0, 11)},
        init_value=0.1),
    ])
    return params