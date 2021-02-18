import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options



dropdown_layout = dcc.Dropdown(
    options=[
        {'label': 'New York city', 'value': 'NYC'},
        {'label': u'Montreal', 'value': 'MTL'},
        {'label': 'San Fransisco', 'value': 'SF'}
    ],
    value='MTL',
    id = 'dropdown'
)
multi_dropdown_layout = dcc.Dropdown(
    options=[
        {'label': 'New York City', 'value': 'NYC'},
        {'label': u'Montréal', 'value': 'MTL'},
        {'label': 'San Francisco', 'value': 'SF'}
    ],
    value=['MTL', 'SF'],
    multi=True
)

radio_layout = dcc.RadioItems(
    options=[
        {'label': 'New York City', 'value': 'NYC'},
        {'label': u'Montréal', 'value': 'MTL'},
        {'label': 'San Francisco', 'value': 'SF'}
    ],
    value='MTL'
)

checkbox_layout = dcc.Checklist(
    options=[
        {'label': 'New York City', 'value': 'NYC'},
        {'label': u'Montréal', 'value': 'MTL'},
        {'label': 'San Francisco', 'value': 'SF'}
    ],
    value=['MTL', 'SF']
)

txt_input_layout = dcc.Input(value='MTL', type='text')

slider_layout = dcc.Slider(
    min=0,
    max=9,
    marks={i: 'Label {}'.format(i) if i == 1 else str(i) for i in range(1, 10)} ,
    value=5
)

app.layout = html.Div([
    html.H6("change the value in the text box to see callbacks in action!"),
    html.Div([
        "input: ",
        dropdown_layout
    ]),
    html.Br(),
    html.Div(id='my-output'),
])

@app.callback(
    # my-output id를 가진 컴포넌트의 children 속성으로 들어간다.
    Output(component_id='my-output', component_property='children'),
    # my-input id 를 가진 컴포넌트의 value 속성을 가져온다.
    Input(component_id='dropdown', component_property='value')
)
def update_output_div(input_value):
    layout = []
    if input_value == 'NYC':
        layout = html.Div([

        html.Label('Multi-Select Dropdown'),
        multi_dropdown_layout,

        html.Label('Radio Items'),
        radio_layout])
    elif input_value == 'MTL':
        layout = html.Div([


        html.Label('Checkboxes'),
        checkbox_layout,

        html.Label('Text Input'),
        txt_input_layout,

        html.Label('Slider'),
        slider_layout
        ], style={'columnCount': 2}
    )
    elif input_value == 'SF':
        layout = html.Div([

        html.Label('Multi-Select Dropdown'),
        multi_dropdown_layout,

        html.Label('Radio Items'),
        radio_layout,

        ], style={'columnCount': 2}
    )

    return layout


if __name__ == '__main__':
    app.run_server(debug=True)


'''
아래 문서를 통해 dash 실습 가능
https://dash.plotly.com/layout
'''
