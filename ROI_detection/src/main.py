# By Rohollah Moosavi Tayebi, email: rohollah.moosavi@uwaterloo.ca/moosavi.tayebi@gmail.com

# This script gets labeled tiles and applies training, validation and testing 
# the model on a new dataset
# It also can be used for production phase to detect Regiong of Interest (ROI) tiles

import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
import numpy as np
import argparse
from data import load_all_images, save_data_list, get_data_loader, generate_fold, load_data_list
from model import get_densenet_model
from learn import train_model, test
from production_test_module import real_test
from production_result_module import real_result
import os
import sys
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

#parameters:
def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", default=25, type=int, required=False, help="batch size")
    parser.add_argument("--num-workers", default=1, type=int, required=False, help="Number of train loader workers.")
    parser.add_argument("--log-interval", default=10, type=int, required=False, help="Training log interval")
    parser.add_argument("--test-portion", default=0.2, type=float, required=False, help="Portion of data which splits dataset to trainset and test-validation set.")
    parser.add_argument("--n-epochs", default=10, type=int, required=False, help="Number of epochs.")
    parser.add_argument("--down-scale", default=2, type=int, required=False, help="Down scaling parameter for image size.")
    parser.add_argument("--n-folds", default=3, type=int, required=False, help="Number of cross validation folds")
    parser.add_argument("--lr", default=1e-4, type=float, required=False, help="learning rate")
    parser.add_argument("--weight-decay", default=5e-4, type=float, required=False, help="weight decay")
    parser.add_argument("--optimizer", default='AdamW', type=str, required=False, help="optimizer can be: AdamW, Adam")
    parser.add_argument("--loss-function", default='CrossEntropyLoss', type=str, required=False, help="loss function can be: CrossEntropyLoss, BCELoss ")
    parser.add_argument("--test-vald-portion", default=0.66, type=float, required=False, help="Portion of data which splits test-validation set to test set and validation set.")
    parser.add_argument("--model-name", default='densenet121', type=str, required=False, help="denseNet model name")
    parser.add_argument("--tile-size", default=512, type=int, required=False, help="Tiles (patches) size (shape")

    parser.add_argument("--threshold", default=None, type=float, required=True, help="threshold")
    parser.add_argument("--data-path", default=None, type=str, required=True, help="dataset path")
    parser.add_argument("--labels-path", default=None, type=str, required=False, help="labels path")
    parser.add_argument("--output-dir", default=None, type=str, required=True, help="output directory")
    parser.add_argument("--load-cache", action="store_true", help="load from cache.")
    parser.add_argument("--test-mode", action="store_true", help="run model only in test mode.")
    parser.add_argument("--predict-mode", action="store_true", help="run model only to get result only.")
    parser.add_argument("--report-excel", action="store_true", help="report test mode result in exel file.")

    parser.add_argument("--r-mean", default=0.4914, type=float, required=False, help="red color mean")
    parser.add_argument("--g-mean", default=0.4822, type=float, required=False, help="green color mean")
    parser.add_argument("--b-mean", default=0.4465, type=float, required=False, help="blue color mean")

    parser.add_argument("--r-std", default=0.2023, type=float, required=False, help="red color standard deviation")
    parser.add_argument("--g-std", default=0.1994, type=float, required=False, help="green color deviation")
    parser.add_argument("--b-std", default=0.2010, type=float, required=False, help="blue color deviation")
    args = parser.parse_args()

    return args

def main(args):
    rgb_mean = (args.r_mean, args.g_mean, args.b_mean)
    rgb_std = (args.r_std, args.g_std, args.b_std)
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation((-2, +2)),
        transforms.ToTensor(),
        transforms.Normalize(rgb_mean, rgb_std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(rgb_mean, rgb_std),
    ])

    if args.load_cache:
        images_list, labels_list, images_list_path = load_data_list(args.data_path + f'/data_{args.down_scale}.pt')
    else:
        images_list, labels_list, images_list_path = load_all_images(args.data_path, args.labels_path, args.down_scale)
        save_data_list(images_list, labels_list, images_list_path, args.data_path+f'/data_{args.down_scale}.pt')

    best_p = []
    best_r = []
    best_a = []
    best_s = []
    best_n = []

    for i, (train_fold_indices, test_fold_indices, validation_fold_indices, images_list_fold, labels_list_fold, images_path_fold) \
            in enumerate(generate_fold(labels_list, args.n_folds, args.test_vald_portion, 'over_sampling', images_list, images_list_path)):


        print(f"\n\nStart of era {i + 1}/{args.n_folds}:")
        train_image_list = [images_list_fold[i] for i in train_fold_indices]
        test_image_list = [images_list_fold[i] for i in test_fold_indices]
        validation_image_list = [images_list_fold[i] for i in validation_fold_indices]

        train_label_list = [labels_list_fold[i] for i in train_fold_indices]
        test_label_list = [labels_list_fold[i] for i in test_fold_indices]
        validation_label_list = [labels_list_fold[i] for i in validation_fold_indices]

        train_path_list = [images_path_fold[i] for i in train_fold_indices]
        test_path_list = [images_path_fold[i] for i in test_fold_indices]
        validation_path_list = [images_path_fold[i] for i in validation_fold_indices]


        train_loader = get_data_loader(
            train_image_list,
            train_label_list,
            train_path_list,
            args.batch_size,
            shuffle=True,
            transform=transform_train,
        )

        test_loader = get_data_loader(
            test_image_list,
            test_label_list,
            test_path_list,
            args.batch_size,
            shuffle=True,
            transform=transform_test
        )

        validation_loader = get_data_loader(
            validation_image_list,
            validation_label_list,
            validation_path_list,
            args.batch_size,
            shuffle=True,
            transform=transform_test
        )

        model = get_densenet_model(args.model_name)

        model = nn.DataParallel(model.to(device))
        if args.optimizer == 'AdamW':
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'Adam':
            pass
        else:
            raise ValueError("Given optimizer is not implemented yet!")

        if args.loss_function == 'CrossEntropyLoss':
            criterion = nn.CrossEntropyLoss()
        elif args.loss_function == 'BCELoss':
            criterion = nn.BCELoss()
        else:
            raise ValueError("Given loss function is not implemented yet!")

        best_acc, best_precision, best_recall, best_specificity, best_NPV = train_model(
            args,
            model,
            train_loader,
            validation_loader,
            args.n_epochs,
            criterion,
            optimizer,
            i,
            args.log_interval,
            device,
            threshold=args.threshold
        )

        best_model = torch.load(args.output_dir + f'/BestModel_{i + 1}.pt')
        best_acc, best_precision, best_recall, best_specificity, best_NPV, test_loss = \
            test(args, best_model, test_loader, criterion, device, i, True, threshold=args.threshold)
        best_a.append(best_acc)
        best_p.append(best_precision)
        best_r.append(best_recall)
        best_s.append(best_specificity)
        best_n.append(best_NPV)
        print("*" * 50)
        print(f"End of era {i + 1}/{args.n_folds}")
        print('-' * 50)
        print("Test dataset result on best chosen model")
        print(f"\t Best Accuracy: {best_acc}")
        print(f"\t Best Precision: {best_precision}")
        print(f"\t Best Recall: {best_recall}")
        print(f"\t Best Specificity: {best_specificity}")
        print(f"\t Best NPV: {best_NPV}")
        print("*" * 50, flush=True)
        best_precision = -1

    print("*" * 50)
    print('-' * 50)
    print(f"\t cross validation Accuracy: {np.mean(best_a)}")
    print(f"\t cross validation Precision (PPV): {np.mean(best_p)}")
    print(f"\t cross validation Recall (Sensitivity): {np.mean(best_r)}")
    print(f"\t cross validation Specificity: {np.mean(best_s)}")
    print(f"\t cross validation NPV: {np.mean(best_s)}")
    print("*" * 50, flush=True)


def production_test(args):
    real_test(args)
def production_result(args):
    real_result(args)

if __name__== "__main__":
    start_time = time.time()

    args = get_args()
    print(f"Code is running. please check the log file in {args.output_dir}/output.log")
    sys.stdout = open(args.output_dir+"/output.log", 'w')
    sys.stderr = open(args.output_dir + "/error.log", 'w')
    print(f"Device is {device}")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}", flush=True)
    if args.test_mode:
        production_test(args)
    elif args.predict_mode:
        production_result(args)
    else:
        main(args)

    exec_time = time.time() - start_time
    print("time: {:02d}m{:02d}s".format(int(exec_time // 60), int(exec_time % 60)))