from bokeh.plotting import figure, show
from bokeh.io import show
from bokeh.models import FileInput

import torch

from PIL import Image

from lab2.trans import test_trans

model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()


def preprocess_image(image_file):
    image = Image.open(image_file)
    transformed_image = test_trans(image)
    return transformed_image.unsqueeze(0)

show(file_input)
show(file_input)

# # prepare some data
# x = [1, 2, 3, 4, 5]
# y = [6, 7, 2, 4, 5]

# # create a new plot with a title and axis labels
# p = figure(title="Simple line example", x_axis_label='x', y_axis_label='y')

# # add a line renderer with legend and line thickness to the plot
# p.line(x, y, legend_label="Temp.", line_width=2)

# # show the results
# show(p)
