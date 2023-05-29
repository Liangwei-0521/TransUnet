import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

from torchvision.transforms import Compose, ToTensor, Normalize, Resize, ToPILImage, PILToTensor
from Data import image_dataset, create_df
from torch.utils.data import DataLoader, Dataset
from Train import Trans_Unet
from flask import Flask, jsonify, request


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


class test:
    def __init__(self, model, path) -> None:
        self.test_model = model
        self.test_model.load_state_dict(torch.load(path))

    def process(self, x):
        # Input the image
        return self.test_model(x)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    # The parameters of down_Conv
    parser.add_argument('--batch_size', type=int, default=16, help='batch')
    parser.add_argument('--image_channels', type=int, default=3, help='initial channel of image')
    parser.add_argument('--height', type=int, default=128, help='the height of image')
    parser.add_argument('--width', type=int, default=256, help='the width of image')
    parser.add_argument('--stride', type=list, default=[(1, 1), (1, 1), (1, 1)], help='convolution stride')
    parser.add_argument('--kernel_size', type=list, default=[(3, 3), (3, 3), (3, 3)], help='convolution kernel size')
    parser.add_argument('--in_channels', type=list, default=[3, 64, 128], help='in channel')
    parser.add_argument('--out_channels', type=list, default=[64, 128, 256], help='out channels')
    parser.add_argument('--num_layers', type=int, default=3, help='the layers of down_Cov')
    parser.add_argument('--pool_kernel', type=int, default=3, help='the max_pool of down_Cov')
    parser.add_argument('--pool_stride', type=int, default=2, help='the  max_pool of down_Cov')
    # The parameters of Encoder
    parser.add_argument('--emb_height', type=int, default=16, help='embedding height')
    parser.add_argument('--emb_width', type=int, default=32, help='embedding width')
    parser.add_argument('--patch_height', type=int, default=2, help='path size')
    parser.add_argument('--patch_width', type=int, default=2, help='path size')
    parser.add_argument('--dim', type=int, default=256, help='embedding dim')
    parser.add_argument('--n_layers', type=int, default=4, help='n_layers')
    parser.add_argument('--n_heads', type=int, default=4, help='n_heads')
    parser.add_argument('--dropout', type=int, default=0.2, help='dropout')
    # The parameters of up_Conv
    parser.add_argument('--up_stride', type=list, default=[(1, 1), (1, 1), (1, 1)], help='up convolution stride')
    parser.add_argument('--up_kernel', type=list, default=[(3, 3), (3, 3), (3, 3)], help='up convolution kernel size')
    parser.add_argument('--up_in_channels', type=list, default=[256, 128, 64], help='up convolution channels')
    parser.add_argument('--up_out_channels', type=list, default=[128, 64, 32], help='up convolution out channels')
    parser.add_argument('--new_height', type=int, default=128, help='the height of new image')
    parser.add_argument('--new_width', type=int, default=256, help='the weight of new image')
    parser.add_argument('--num_classes', type=int, default=24, help='the classes of pixel')
    args = parser.parse_args()
    # The image of testing set
    df_list = create_df(image_path='./Image_2/val/val-org-img/', label_path='./Image_2/val/val-label-img/')
    test_set = image_dataset(image_path=r'./Image_2/val/val-org-img/', label_path=r'./Image_2/val/val-label-img/',
                             x=df_list, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    t_ = test(model=Trans_Unet(args=args), path=r'./Result/model.pth')

    # for i, (images, labels) in enumerate(test_loader):
    #     predict = t_.process(images)
    #     p = show(image=torch.argmax(predict[0, :, :, :], dim=0).detach().cpu().numpy(), num_class=24)
    #     l = show(image=labels[0, :, :].detach().cpu().numpy(), num_class=24)
    #     fig = plt.figure()
    #     ax1 = plt.subplot(1, 2, 1)
    #     ax1.imshow(p)
    #     ax2 = plt.subplot(1, 2, 2)
    #     ax2.imshow(l)
    #     plt.tight_layout()
    #     plt.show()
    #     print(torch.argmax(predict[0, :, :, :], dim=0).shape)
    #     print(torch.argmax(predict[0, :, :, :], dim=0), '\n', labels[0, :, :])

    def read_image(img):
        transform_image = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                   Resize((128, 256))])
        img = transform_image(cv2.imread(img)).unsqueeze(dim=0)
        return img

    # Web Application
    app = Flask(__name__)

    @app.route('/predict', methods=['POST'])
    def predict():
        if request.method == 'POST':
            # 输入file
            file = request.files['file']
            image = file.read()
            predict = t_(read_image(img=image))
            return jsonify(
                {
                    'predict': predict
                })
    app.run()
