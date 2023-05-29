import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, ToPILImage, PILToTensor
from TransUnet.Data import image_dataset, create_df
from torch.utils.data import DataLoader, Dataset
from Unet.unet import U_net


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


class model_test:
    def __init__(self, model, path) -> None:
        self.test_model = model
        self.test_model.load_state_dict(torch.load(path))

    def process(self, x):
        # Input the image
        return self.test_model(x)


if __name__ == '__main__':
    # The data
    df_list = create_df(image_path='../TransUnet/Image_2/val/val-org-img/',
                        label_path='../TransUnet/Image_2/val/val-label-img/')
    test_set = image_dataset(image_path=r'../TransUnet/Image_2/val/val-org-img/',
                             label_path=r'../TransUnet/Image_2/val/val-label-img/',
                             x=df_list,
                             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    # The model
    t_ = model_test(model=U_net(n_channels=3, n_classes=24), path=r'./result/unet_model.pth')

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
        transform_image = Compose([ToTensor(),
                                   Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                   Resize((128, 256))])
        img = transform_image(cv2.imread(img)).unsqueeze(dim=0)
        return img


    # 测试
    image_t = read_image(r'../TransUnet/Image_2/val/val-org-img/' + df_list[0] + '.jpg')
    predict = t_.process(image_t)
    p = show(image=torch.argmax(predict[0, :, :, :], dim=0).detach().cpu().numpy(), num_class=24)
    plt.imshow(p)
    plt.show()
