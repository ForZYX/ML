from torch.utils.data import Dataset
import pandas as pd
import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder




class MyDataset(Dataset):
    def __init__(self, folder='F://data//classify-leaves', train=True, transform=None):
        super(MyDataset, self).__init__()
        csv_file = 'train.csv' if train else 'test.csv'
        self.root = folder
        self.train_set = pd.read_csv(os.path.join(folder, csv_file))
        self.transform = transform
        self.labels = pd.read_csv(os.path.join(folder, 'train.csv'))['label'].unique()
        self.le = LabelEncoder()
        self.le.fit(self.labels)

    def load_data(self, path):
        return (cv2.imread(path).transpose(2, 0, 1)).astype(np.float32) / 255

    def __getitem__(self, index):
        img_path, label = self.train_set.iloc[index, :]
        img_path = os.path.join(self.root, img_path)
        img = self.load_data(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.trans_from_str2int([label])[0]

    def __len__(self):
        return self.train_set.shape[0]

    def trans_from_str2int(self, strs):
        return self.le.transform(strs)


if __name__ == '__main__':
    d = MyDataset()
    print(d.__getitem__(0))
