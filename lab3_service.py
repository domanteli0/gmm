from flask import Flask, jsonify, request
import numpy as np
from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from PIL import Image
import torch
from lab3.show import mask_to_rgb_image
from lab3.classes import classes
from lab1.device import device

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
DIM=128

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

from lab3.net import Net
def load_model(path = "lab3/model_attempt1.pth"):
    model = torch.load(path)
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

from lab3.trans import test_trans as trans

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

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        if file and allowed_file(file.filename):
            # Save uploaded file
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            # Process uploaded image
            output_paths = process_image(file_path, app.config['OUTPUT_FOLDER'])
            return render_template('result.html', original=filename, outputs=output_paths)
    return render_template('./index.html')

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

if __name__ == '__main__':
    app.run(debug=True)