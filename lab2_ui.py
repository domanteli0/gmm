# from bokeh.plotting import figure, show
import base64
import io
from bokeh.io import show
from lab2.model import Model

import torch

from PIL import Image

from lab2.trans import test_trans

def load_model(path = "lab2.pth"):
    model = Model()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def preprocess_image(image_as_base64_str):
    image_as_base64 = base64.b64decode(image_as_base64_str)
    image_buffer = io.BytesIO(image_as_base64)
    image = Image.open(image_buffer)
    image = test_trans(image)
    return image.unsqueeze(0)

# model = load_model()
# user_image = FileInput()

# show(user_image)

# prediction = model(preprocess_image(user_image))

# show(prediction)
from random import random

from bokeh.layouts import column
from bokeh.models import Button
from bokeh.palettes import RdYlBu3
from bokeh.plotting import figure, curdoc
from bokeh.models import FileInput

# create a plot and style its properties
p = figure(x_range=(0, 100), y_range=(0, 100), toolbar_location=None)
p.border_fill_color = 'black'
p.background_fill_color = 'black'
p.outline_line_color = None
p.grid.grid_line_color = None

# add a text renderer to the plot (no data yet)
r = p.text(x=[], y=[], text=[], text_color=[], text_font_size="26px",
           text_baseline="middle", text_align="center")

i = 0

ds = r.data_source

# create a callback that adds a number in a random location
def callback():
    global i

    # BEST PRACTICE --- update .data in one step with a new dict
    new_data = dict()
    new_data['x'] = ds.data['x'] + [random()*70 + 15]
    new_data['y'] = ds.data['y'] + [random()*70 + 15]
    new_data['text_color'] = ds.data['text_color'] + [RdYlBu3[i%3]]
    new_data['text'] = ds.data['text'] + [str(i)]
    ds.data = new_data

    i = i + 1

# add a button widget and configure with the call back
button = Button(label="Press Me")
button.on_event('button_click', callback)

model = load_model()

def user_image(attr, o, new):
    # print(f" . type {type(new)}")
    # print(f"  new {new}")
    prediction = model(preprocess_image(new))
    print(f"  pred {prediction}")
    # show(new)

    

image_path = FileInput(accept = '.jpg')
image_path.on_change('value', user_image)

# put the button and plot in a layout and add to the document
curdoc().add_root(image_path)
