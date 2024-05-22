import numpy as np
from flask import Flask, request, render_template, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import os
import cv2

import lab3.trans
from lab3.classes import classes
from lab1.device import device
from lab3.show import mask_to_rgb_image, save_masks_nouveau
import torch
from torchvision import transforms
from PIL import Image
from lab1.device import device

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

from lab3.net import DIM, Net
from lab3.trans import test_trans as trans

def load_model(path = "lab3/save/net_attempt5-91.pt"):
    model = torch.load(path)
    print(f"TYPE {type(model)}")

    model.eval()
    return model

model = load_model()

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_mask(mask):
    return mask_to_rgb_image(mask.permute(2, 1, 0), classes = classes)

def upscale_and_smooth(img, size, sigma=3):
    # Upscale the binary image
    upscaled_image = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)

    # Apply Gaussian blur for smoothing
    # smoothed_image = cv2.GaussianBlur(upscaled_image, (0, 0), sigmaX=sigma, sigmaY=sigma)

    return upscaled_image

def render_mask(mask):
    shape = mask.shape[0:2]
    print("Rendering mask")
    # Define colors for the mask and background
    mask_color = (255, 255, 255, 255)  # Red for the mask
    background_color = (0, 0, 0, 0)  # White for the background

    # Create a blank image with the same shape as the mask array

    print(f"mask.shape: {mask.shape}")
    print(f"shape: {shape}")
    image = Image.new('RGBA', shape, background_color)
    pixels = image.load()
    print("Pixels loaded")

    # Iterate through the mask array and set corresponding pixels in the image
    for y in range(shape[0]):
        for x in range(shape[1]):
            if mask[y, x] == 1:
                pixels[x, y] = mask_color
    return image

def process_image(input_path, output_folder):
    img = trans(Image.open(input_path).convert('RGB')).unsqueeze(0).to(device)
    original_dims = img.shape[0:2][::-1]
    masks = model(img/255.).detach().cpu()
    print("--------")
    print(f"shape1: {masks.shape}")
    print(f"shape2: {masks[0].shape}")
    mask = process_mask(masks[0])
    print(mask.shape)
    paths = []

    print(mask.shape, original_dims)
    # mask = upscale_and_smooth(mask, original_dims)
    img = render_mask(mask)
    path = os.path.join(output_folder, f'mask.png')
    img.save(path)
    paths.append(path)

    return paths

def to_pil_image(image_as_base64_str):
    import base64
    import io
    image_as_base64 = base64.b64decode(image_as_base64_str)
    image_buffer = io.BytesIO(image_as_base64)
    image = Image.open(image_buffer)
    return image

from lab3.trans import test_trans
def preprocess_image(image_as_base64_str):
    image = to_pil_image(image_as_base64_str).convert('RGB')
    image = test_trans(image)
    return image.unsqueeze(0)

def transform_image(image_bytes):
    transform = lab3.trans.test_trans
    image = Image.open(image_bytes).convert('RGB')
    return transform(image).unsqueeze(0)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes).to(device)
    outputs = model(tensor).cpu().detach().squeeze(0)
    return outputs

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            raise 'No file part'
        file = request.files['file']
        if file.filename == '':
            raise 'No selected file'
        if not allowed_file(file.filename):
            raise 'Invalid file type'
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(f"{app.config['UPLOAD_FOLDER']}/{filename}")

            with open(f"{app.config['UPLOAD_FOLDER']}/{filename}", 'rb') as f:
                prediction = get_prediction(f)

            prediction_image_path = f"{app.config['OUTPUT_FOLDER']}/prediction.png"
            save_masks_nouveau(prediction, classes, prediction_image_path)

            return send_from_directory(app.config['OUTPUT_FOLDER'], 'prediction.png')

    return '''
    <!doctype html>
    <title>Upload an Image</title>
    <h1>Upload an Image for Segmentation</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

if __name__ == '__main__':
    app.run(debug=True, port=5002)
