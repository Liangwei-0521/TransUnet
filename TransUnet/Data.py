import os 
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, ToPILImage, PILToTensor


def create_df(image_path, label_path):

    train_image, train_label = [], []
    for name in os.listdir(image_path):
        train_image.append(name.split('.')[0])

    for name in os.listdir(label_path):
        train_label.append(name.split('.')[0])

    # 交集
    train_df = list(set(train_image) & set(train_label))
    return train_df


class image_dataset(Dataset):
    def __init__(self, image_path, label_path, x, mean, std) -> None:
        super().__init__()
        self.img_path = image_path
        self.label_path = label_path
        self.x = x
        self.mean = mean
        self.std = std
        self.transform_image = Compose([ToTensor(), Normalize(self.mean, self.std), Resize((128, 256))])
        self.transform_label = Compose([ToTensor(), ToPILImage(), PILToTensor(), Resize((128, 256))])

    def __len__(self, ):
        return len(self.x)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path + self.x[idx] + '.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = cv2.imread(self.label_path + self.x[idx] + '.png', cv2.IMREAD_GRAYSCALE)  # cv2.IMREAD_GRAYSCALE np.ndaary()
        img = self.transform_image(img)
        label = self.transform_label(label).squeeze(0)
        return img, label
    

if __name__ == '__main__':

    df_list = create_df(image_path='Image_2/train/train-org-img/', label_path='Image_2/train/train-label-img/')
    assert len(df_list) != 0
    train_set = image_dataset(image_path=r'./Image_2/train/train-org-img/', label_path=r'./Image_2/train/train-label-img/',
                              x=df_list,
                              mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_loader = DataLoader(train_set, batch_size=16, num_workers=2, shuffle=True)

    for i, (image, label) in enumerate(train_loader):
        print('Image:', image.shape, '\n', 'Label:', label)


