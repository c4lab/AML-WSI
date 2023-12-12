# By Rohollah Moosavi Tayebi, email: rohollah.moosavi@uwaterloo.ca/moosavi.tayebi@gmail.com

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pandas as pd
from PIL import Image, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from skimage.transform import rescale, resize, downscale_local_mean

# Train on balanced data and test on imbalanced data:
class Dataset(Dataset):
    def __init__(self, image_list, labels_list, images_list_path, transform=None):
        self.labels = torch.tensor(labels_list)
        self.image_list = image_list
        self.path = images_list_path
        self.transform = transform

    def __getitem__(self, idx):
        x = self.image_list[idx]
        if self.transform:
            x = self.transform(x)
        else:
            x = torch.tensor(x)
        x = torch.tensor(x).cpu()
        y = torch.tensor(self.labels[idx]).long()  # For cross entropy
        # p = torch.tensor(self.path[idx])
        p = self.path[idx]
        return x, y, p

    def __len__(self):
        return self.labels.shape[0]

def load_all_images(images_path, labels_path, down_scale):
    labels = pd.read_csv(labels_path, sep="\t")

    labels_dict = {}
    imagesListPath = []
    labelsList = []
    image_list = []
    for i in range(labels.shape[0]):
        f_name = labels.iloc[i, 0]
        label = labels.iloc[i, 1]
        labels_dict[f_name] = label

    for path, subdirs, files in os.walk(images_path):
        for f_name in files:
            if f_name[:-4] in labels_dict:
                imagesListPath.append(path + "/" + f_name)
                labelsList.append(labels_dict[f_name[:-4]])

    shuffle_idx = np.random.permutation(len(labelsList))
    imagesListPath = [imagesListPath[i] for i in shuffle_idx]
    labelsList = [labelsList[i] for i in shuffle_idx]

    for i, path in tqdm(enumerate(imagesListPath)):
        x = plt.imread(path)
        x = resize(x, (x.shape[0] // down_scale, x.shape[1] // down_scale, 3))
        image_list.append(Image.fromarray(np.uint8(x * 255)))

    return image_list, labelsList, imagesListPath

def get_data_loader(images_list, labels_list, images_list_path, batch_size, shuffle, transform=None):
    dataset = Dataset(images_list, labels_list, images_list_path, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader

def save_data_list(images_list, labels_list, images_list_path, path):
    file = open(path, "wb" )
    pickle.dump((images_list, labels_list, images_list_path), file)
    file.close()

def load_data_list(path):
    file = open(path, "rb" )
    images_list, labels_list, images_list_path = pickle.load(file)
    file.close()
    return images_list, labels_list, images_list_path

def generate_fold(labels_list, fold, test_vald_portion, imbalanced_handler, images_list, images_list_path):
    labels = np.array(labels_list)
    pi = np.where(labels == 1)[0]
    ni = np.where(labels == 0)[0]
    portion = ni.shape[0] / pi.shape[0]

    for i in range(fold):
        positive_test_vald = pi[int(pi.shape[0] * (i / fold)):int(pi.shape[0] * ((i + 1) / fold))]
        negative_test_vald = ni[int(positive_test_vald.shape[0] * portion * i):int(positive_test_vald.shape[0] * portion * (i + 1))]

        positive_vald = positive_test_vald[int(test_vald_portion * positive_test_vald.shape[0]):]
        negative_vald = negative_test_vald[int(test_vald_portion * negative_test_vald.shape[0]):]

        positive_test = positive_test_vald[:int(test_vald_portion * positive_test_vald.shape[0])]
        negative_test = negative_test_vald[:int(test_vald_portion * negative_test_vald.shape[0])]

        positive_train = []
        negative_train = []
        for j in pi:
            if j not in positive_test_vald:
                positive_train.append(j)

        for j in ni:
            if j not in negative_test_vald:
                negative_train.append(j)

        if imbalanced_handler == 'under_sampling':
            idxs = np.random.choice(len(negative_train), size=len(positive_train), replace=False)
            tmp = [negative_train[j] for j in idxs]
            negative_train = tmp
        elif imbalanced_handler == 'over_sampling':
            temp_positive_train = list(positive_train)  #call by local
            for i in positive_train:
                image = images_list[i]
                last = len(images_list)

                hor_flip = image.transpose(Image.FLIP_LEFT_RIGHT)
                ver_flip = image.transpose(Image.FLIP_TOP_BOTTOM)

                enhancer = ImageEnhance.Contrast(image)
                contrast1 = enhancer.enhance(0.8)
                contrast2 = enhancer.enhance(1.2)
                gaus = image.filter(ImageFilter.GaussianBlur(radius=1.2))
                images_list.append(hor_flip)
                images_list_path.append(images_list_path[i])
                temp_positive_train.append(last)
                labels_list.append(1)
                last += 1

                images_list.append(ver_flip)
                images_list_path.append(images_list_path[i])
                temp_positive_train.append(last)
                labels_list.append(1)
                last += 1

                images_list.append(contrast1)
                images_list_path.append(images_list_path[i])
                temp_positive_train.append(last)
                labels_list.append(1)
                last += 1

                images_list.append(contrast2)
                images_list_path.append(images_list_path[i])
                temp_positive_train.append(last)
                labels_list.append(1)
                last += 1

                images_list.append(gaus)
                images_list_path.append(images_list_path[i])
                temp_positive_train.append(last)
                labels_list.append(1)
                last += 1

            positive_train = temp_positive_train
            idxs = np.random.choice(len(positive_train), size=len(negative_train), replace=True)
            tmp = [positive_train[j] for j in idxs]
            positive_train = tmp
        else:
            raise ValueError("Error passing imbalanced_handler parameter! ")

        data_train = np.array(positive_train + negative_train)
        data_test = np.concatenate((positive_test, negative_test))
        data_validation = np.concatenate((positive_vald, negative_vald))

        yield data_train, data_test, data_validation, images_list, labels_list, images_list_path