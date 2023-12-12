# By Rohollah Moosavi Tayebi, email: rohollah.moosavi@uwaterloo.ca/moosavi.tayebi@gmail.com

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from xlwt import Workbook
import tqdm
import numpy as np
from model import get_densenet_model
import os
import sys
from collections import OrderedDict
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
device = "cuda" if torch.cuda.is_available() else "cpu"
from utils import copy_files

class Dataset(Dataset):
    def __init__(self, image_list, images_list_path, transform=None):
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
        p = self.path[idx]
        return x, p

    def __len__(self):
        return len(self.image_list)


def load_all_images_production_result(input_tile_size, images_path, down_scale):
    imagesListPath = []
    image_list = []

    for path, subdirs, files in os.walk(images_path):
        for f_name in files:
            file_path = path + "/" + f_name
            x = plt.imread(file_path)

            if x.shape[0] == input_tile_size and x.shape[1] == input_tile_size:
                x = resize(x, (x.shape[0] // down_scale, x.shape[1] // down_scale, 3))
                y = Image.fromarray(np.uint8(x * 255))
                image_list.append(Image.fromarray(np.uint8(x * 255)))
                imagesListPath.append(file_path)

    return image_list, imagesListPath

def load_data_list_production_result(path):
    file = open(path, "rb" )
    images_list, images_list_path = pickle.load(file)
    file.close()
    return images_list, images_list_path

def save_data_list_production_result(images_list, images_list_path, path):
    file = open(path, "wb" )
    pickle.dump((images_list, images_list_path), file)
    file.close()

def get_data_loader_production_result(images_list, images_list_path, batch_size, shuffle, transform=None):
    dataset = Dataset(images_list, images_list_path, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader

def real_result(args):
    rgb_mean = (args.r_mean, args.g_mean, args.b_mean)
    rgb_std = (args.r_std, args.g_std, args.b_std)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(rgb_mean, rgb_std),
    ])

    print("loading data...")
    if args.load_cache:
        test_image_list, test_path_list = load_data_list_production_result(args.data_path + f'\\real_data_predict_{args.down_scale}.pt')
    else:
        test_image_list, test_path_list = load_all_images_production_result(args.tile_size , args.data_path, args.down_scale)
        # save_data_list_production_result(test_image_list, test_path_list, args.data_path + f'\\real_data_predict_{args.down_scale}.pt')

    test_loader = get_data_loader_production_result(
        test_image_list,
        test_path_list,
        args.batch_size,
        shuffle=False,
        transform=transform_test
    )
    model_path = "../model"
    """
    best_model = torch.load(weight_path + '/BestModel.pt')
    torch.save(best_model.state_dict(), args.output_dir + '/BestModel_Dict.pt')
    """
    model_name = args.model_name
    best_model = get_densenet_model(model_name)

    # temp_dict = torch.load(args.output_dir + '/BestModel_Dict.pt')
    temp_dict = torch.load(model_path+'/BestModel.pt')
    new_temp_dict = OrderedDict()

    for key, value in temp_dict.items():
        new_key = key[7:]
        new_temp_dict[new_key] = value

    best_model.load_state_dict(new_temp_dict)
    best_model1 = best_model.to(device)

    best_model1.eval()
    z_list = []
    with torch.no_grad():
        for i, (x, z) in enumerate(test_loader):
            x = x.to(device) 
            output = best_model1(x)
            if args.threshold == None:
                pred = torch.argmax(output, dim=1)
            else:
                soft = F.softmax(output)
                pred = torch.zeros(output.shape[0])
                for j in range(pred.shape[0]):
                    if soft[j, 1] > args.threshold:
                        pred[j] = 1
                    else:
                        pred[j] = 0

            if i == 0:
                pred_tot = pred.cpu()
            else:
                pred_tot = torch.cat((pred_tot.cpu(), pred.cpu()), dim=0)

            for p in z:
                z_list.append(p)

    pos = 0
    neg = 0

    pos_list = []
    neg_list = []

    for i in range(pred_tot.shape[0]):
        if pred_tot[i] == 0:
            neg += 1
            neg_list.append(z_list[i])
        else:
            pos += 1
            pos_list.append(z_list[i])

    if args.report_excel:
        wb = Workbook()

        sheetNumber0 = wb.add_sheet('Sheet 0')
        sheetNumber1 = wb.add_sheet('Sheet 1')

        row = 0
        sheetNumber0.write(row, 0, 'Posivitve List')
        sheetNumber0.write(row, 1, pos)
        row += 1
        for i in pos_list:
            sheetNumber0.write(row, 0, i)
            row += 1

        row = 0
        sheetNumber1.write(row, 0, 'Negative List')
        sheetNumber1.write(row, 1, neg)
        row += 1
        for i in neg_list:
            sheetNumber1.write(row, 0, i)
            row += 1

        wb.save(args.output_dir +'/porduction_result.xls')

    print(f"Positive (ROI): {pos}")
    print(f"Negative (Non-ROI): {neg}")

    # copy_files(args)