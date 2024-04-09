from bokeh.plotting import figure, curdoc
from bokeh.models import FileInput, Paragraph
from bokeh.layouts import column, row, layout
from bokeh.client import push_session
import numpy as np

import torch

from PIL import Image

from lab2.model1 import Model
def load_model(path = "model2_attempt5.pth"):
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

from lab2.trans1 import test_trans
def preprocess_image(image_as_base64_str):
    image = to_pil_image(image_as_base64_str).convert('RGB')
    image = test_trans(image)
    return image.unsqueeze(0)

model = load_model()
class_names = [ "airplane", "boat", "bus", "train" ]

fig = figure()

prediction_p = Paragraph(text = "Upload an image", width=250)

def user_image(attr, o, new):
    global prediction_p
    global fig
    confidence, label = model(preprocess_image(new)).softmax(dim = 1).max(dim = 1)
    confidence = confidence[0]
    class_name = class_names[label[0]]
    text = f"Prediction: {class_name} [Confidence: {confidence * 100:.1f}%]"
    print(text)

    prediction_p.text = text
    fig.image_rgba(image=[np.array(to_pil_image(new))])

image_path = FileInput(accept = '.jpg')
image_path.on_change('value', user_image)

# put the button and plot in a layout and add to the document
curdoc().add_root(layout([[column([image_path, prediction_p]), fig]]))
# session = push_session(curdoc())
# session.show()

