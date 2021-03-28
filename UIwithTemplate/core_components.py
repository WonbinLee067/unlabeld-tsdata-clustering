import dash_core_components as dcc
import dash_daq as daq
def tab_layout(id, init_value, children):
    return dcc.Tabs(id=id, value=init_value,
        children = [dcc.Tab(label=lbl, value=val) for lbl, val in children.items()]
    )

def dropdown_layout(id, options, init_value):
    return dcc.Dropdown(
        options = options,
        value=init_value,
        id=id )
def multi_dropdown_layout(id, options, init_value):
    return dcc.Dropdown(
        options = options,
        value = init_value,
        id = id,
        multi = True )

def radio_layout(id, options, init_value):
    return dcc.RadioItems(
        options=options,
        value=init_value,
        id = id )

def checkbox_layout(id, options, init_value):
    return dcc.Checklist(
        options=options,
        value=init_value,
        id = id )

def txt_input_layout(id, init_value='', input_type='text', placeholder='input value'):
    return  dcc.Input(value=init_value, type=input_type, id=id, placeholder=placeholder) # 더 알아봐서 추가

def num_input_layout(id, min=0, max=50, init_value=1, placeholder=""):
    return dcc.Input(value=init_value, id=id, type='number', min=min, max=max)

def slider_layout(id, min, max, marks,init_value=1, step=1):
    return dcc.Slider(
        min=min, 
        max=max, 
        marks=marks ,
        value=init_value, 
        id=id,
        step=step)

def switch_layout(id, on=False, label=""):
    return daq.BooleanSwitch(
        id = id,
        on = on,
        label = label,
        labelPosition="top"
    )