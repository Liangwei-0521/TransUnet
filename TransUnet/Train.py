import numpy as np
import matplotlib.pyplot as plt
from Data import image_dataset, create_df
from torch.utils.data import DataLoader, Dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from Model.Conv import Embeddings
from Model.Encoder import Trans_Encoder
from Model.Conv import Cascaded_Upsampler


def pixel_accuracy(output, mask, device):
    # Pixel Accuracy
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1).to(device)
        correct = torch.eq(output, mask.to(device)).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy


def dice_loss(y_pred, y_true):
    # convert y_pred and y_true to float tensors
    y_pred = y_pred.float()
    y_true = y_true.float()
    # flatten y_pred and y_true
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    # compute the intersection and union
    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum()
    # compute the dice coefficient
    dice = 2 * intersection / (union + 1e-7)
    # return the dice loss
    return 1 - dice


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


class Trans_Unet(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.down_conv = Embeddings(args=args)
        self.transformer = Trans_Encoder(emb_height=args.emb_height, emb_width=args.emb_width, args=args)
        self.up_conv = Cascaded_Upsampler(args=args)

    def forward(self, x):
        x, content = self.down_conv(x)
        x_ = self.transformer(x)
        result = self.up_conv(x_, content)
        return result


class train:
    def __init__(self, learning_rate, device) -> None:
        self.device = device
        self.trans_unet = Trans_Unet(args=args).to(device)
        self.adam = Adam(self.trans_unet.parameters(), lr=learning_rate)
        self.loss = torch.nn.CrossEntropyLoss()

    def process(self, x, epoches):
        # x : The dataLoader of training set
        all_loss = []
        for epoch in range(epoches):
            print('Episode:', epoch)
            item_loss = 0
            for i, (image, label) in enumerate(x):
                # training process
                output = self.trans_unet(image.to(self.device))
                loss = self.loss(output, label.type(torch.LongTensor).to(self.device)) + dice_loss(
                    torch.argmax(output, dim=1), label.to(self.device))
                self.adam.zero_grad()
                loss.backward()
                self.adam.step()
                accuracy = pixel_accuracy(output, label, device='cuda:0' if torch.cuda.is_available() else 'cpu')
                item_loss = item_loss + loss.item()
                print('---Accuracy---{}, ---Loss---{}'.format(accuracy, loss.item()))
                if i % 10 == 0:
                    # Output the predicted labels for each pixel
                    print(torch.argmax(output[0, :, :, :], dim=0), '\n', label[0, :, :])
                    predict = show(image=torch.argmax(output[0, :, :, :], dim=0).detach().cpu().numpy(), num_class=24)
                    label = show(image=label[0, :, :].detach().cpu().numpy(), num_class=24)
                    fig = plt.figure(figsize=(10, 3))
                    ax1 = plt.subplot(1, 2, 1)
                    ax1.set_title("Predict")
                    ax1.imshow(predict)
                    ax2 = plt.subplot(1, 2, 2)
                    ax2.set_title("Label")
                    ax2.imshow(label)
                    plt.tight_layout()
                    plt.show()
            all_loss.append(item_loss)
        # 模型保存
        path = './Result/model.pth'
        torch.save(self.trans_unet.state_dict(), path, _use_new_zipfile_serialization=False)
        # 可视化损失
        plt.title('Loss')
        plt.plot(range(len(all_loss)), all_loss)
        plt.show()


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
    # The image of training set
    df_list = create_df(image_path='./Image_2/train/train-org-img/', label_path='./Image_2/train/train-label-img/')
    assert len(df_list) != 0
    train_set = image_dataset(image_path=r'./Image_2/train/train-org-img/',
                              label_path=r'./Image_2/train/train-label-img/',
                              x=df_list, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_loader = DataLoader(train_set, batch_size=16, shuffle=False)
    # TransUnet
    model_train = train(learning_rate=0.0001, device='cuda:0' if torch.cuda.is_available() else 'cpu')
    model_train.trans_unet.load_state_dict(torch.load(r'./Result/model.pth'))
    model_train.process(x=train_loader, epoches=2)
