import dash_core_components as dcc
import dash_html_components as html
import core_components as cc


def recurrence_plot():
    params = html.Div([
            dcc.Store(id='store-rp-param', data=[]),
            html.H4("RP Parameters"),
            html.H6("Dimension"),
            html.Label("RP 궤적의 차원수를 결정한다. 공간 궤적 좌표 생성에 쓰이는 데이터 개수이다."),
            cc.num_input_layout('dimension', init_value=1, min=1, placeholder="dimension"),

            html.H6("Time-Delay"),
            html.Label("공간 궤적 좌표 생성시 사용되는 기존 좌표 데이터의 시간 차이를 뜻한다. 따라서 1dim 데이터 사용시 큰 의미가 없다."),
            cc.num_input_layout('time-delay', init_value=1, min=1, placeholder="time delay"),

            html.H6("Threshold"),
            html.Label("궤적의 거리 최솟값을 설정한다."),
            cc.dropdown_layout('threshold',
            [
                {'label': 'float', 'value':'F'},
                {'label': 'point', 'value':'P'},
                {'label': 'distance', 'value':'D'},
                {'label': 'None', 'value' : 'N'}
            ], 'F'),
            html.Label("percentage if point or distance"),
            cc.slider_layout('percentage', min=10, max=60, marks={i: '{}'.format(i) for i in range(10, 61, 10)})
        ])
    return params

def raw_img():
    params = html.Div([
            dcc.Store(id='store-raw-img-param', data=[]),
            html.H4("RAW Data Preprocessing"),
            cc.num_input_layout('raw-img-sample', init_value=1, min=1, placeholder="sample"),
        ])
    return params
