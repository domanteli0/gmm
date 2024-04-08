from bokeh.plotting import figure, curdoc
from bokeh.models import FileInput
from bokeh.layouts import column
import numpy as np

import torch

from PIL import Image

from lab2.trans import test_trans

from lab2.model1 import Model
def load_model(path = "model1_attempt3.pth"):
    model = Model()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def to_pil_image(image_as_base64_str):
    import base64
    import io
    image_as_base64 = base64.b64decode(image_as_base64_str)
    image_buffer = io.BytesIO(image_as_base64)
    image = Image.open(image_buffer)
    return image

def preprocess_image(image_as_base64_str):
    image = to_pil_image(image_as_base64_str).convert('RGB')
    image = test_trans(image)
    return image.unsqueeze(0)

# from random import random

# from bokeh.models import Button
# from bokeh.palettes import RdYlBu3

# # create a plot and style its properties
# p = figure(x_range=(0, 100), y_range=(0, 100), toolbar_location=None)
# p.border_fill_color = 'black'
# p.background_fill_color = 'black'
# p.outline_line_color = None
# p.grid.grid_line_color = None

# # add a text renderer to the plot (no data yet)
# r = p.text(x=[], y=[], text=[], text_color=[], text_font_size="26px",
#            text_baseline="middle", text_align="center")

# i = 0

# ds = r.data_source

# # create a callback that adds a number in a random location
# def callback():
#     global i

#     # BEST PRACTICE --- update .data in one step with a new dict
#     new_data = dict()
#     new_data['x'] = ds.data['x'] + [random()*70 + 15]
#     new_data['y'] = ds.data['y'] + [random()*70 + 15]
#     new_data['text_color'] = ds.data['text_color'] + [RdYlBu3[i%3]]
#     new_data['text'] = ds.data['text'] + [str(i)]
#     ds.data = new_data

#     i = i + 1

# # add a button widget and configure with the call back
# button = Button(label="Press Me")
# button.on_event('button_click', callback)

model = load_model()
class_names = [ "airplane", "boat", "bus", "train" ]

placeholder_img = Image.open('lab2/transparent.jpeg')
fig = figure(x_range=(0, 1), y_range=(0, 1))
fig.image_rgba(image=[np.array(placeholder_img)])

# upload_button = figure(x_range = (1, 2), y_range = (1, 2))

def user_image(attr, o, new):
    confidence, label = model(preprocess_image(new)).softmax(dim = 1).max(dim = 1)
    print(f"  conf  {confidence[0]}")
    print(f"  label {class_names[label[0]]}")

image_path = FileInput(accept = '.jpg')
image_path.on_change('value', user_image)

# put the button and plot in a layout and add to the document
curdoc().add_root(column(image_path, fig))
