import torch
import h5py
import numpy as np
from PIL import Image

class etl(torch.utils.data.Dataset):

    def __init__(self, split, transform=None):
        self.transform = transform
        self.split = split  # training set or test set
        self.data = h5py.File('etl.h5', 'r', driver='core')

        if self.split == 'training':
            self.train_datas = self.data['training_pixel']
            self.train_labels = self.data['training_label']
            self.train_datas = np.asarray(self.train_datas)
        else:
            self.test_datas = self.data['testing_pixel']
            self.test_labels = self.data['testing_label']
            self.test_datas = np.asarray(self.test_datas)

    def __getitem__(self, index):

        if self.split == 'training':
            img, target = self.train_datas[index], self.train_labels[index]
        else:
            img, target = self.test_datas[index], self.test_labels[index]


        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        img = img[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        if self.split == 'training':
            return len(self.train_datas)
        else:
            return len(self.test_datas)