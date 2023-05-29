import io
import cv2
import flask
import requests
import torch
import numpy as np
from flask import Flask, request, render_template
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, ToPILImage, PILToTensor
from Unet.test import model_test
from Unet.unet import U_net

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


U_model = model_test(model=U_net(n_channels=3, n_classes=24), path=r'../../Unet/result/unet_model.pth')


def read_image(img):
    transform_image = Compose([ToTensor(),
                               Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                               Resize((128, 256))])
    img = transform_image(cv2.imread(img)).unsqueeze(dim=0)

    return img


def show(image, num_class):
    label_colors = np.array([
        (0, 0, 0),  # unlabeled
        (128, 64, 128),  # paved-area
        (130, 76, 0),  # dirt
        (0, 102, 0),  # grass
        (112, 103, 87),  # gravel
        (28, 42, 168),  # water
        (48, 41, 30),  # rocks
        (0, 50, 89),  # pool
        (107, 142, 35),  # vegetation
        (70, 70, 70),  # roof
        (102, 102, 156),  # wall
        (254, 228, 12),  # window
        (254, 148, 12),  # door
        (190, 153, 153),  # fence
        (153, 153, 153),  # fence-pole
        (255, 22, 96),  # person
        (102, 51, 0),  # dog
        (9, 143, 150),  # car
        (119, 11, 32),  # bicycle
        (51, 51, 0),  # tree
        (190, 250, 190),  # bald-tree
        (112, 150, 146),  # art-marker
        (2, 135, 115),  # obstacle
        (255, 0, 0),  # conflicting
    ])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, num_class):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb


@app.route('/predict', methods=['POST'])
def predict():
    data = {
        'success': False
    }
    if flask.request.method == 'POST':
        if flask.request.files.get("image"):
            image = flask.request.files['image'].read()
            image = image.open(io.BytesIO(image))
            predict = U_model.process(read_image(image))
            data['predictions'] = show(predict, num_class=24)
            return flask.jsonify(data)


if __name__ == '__main__':
    app.run()
