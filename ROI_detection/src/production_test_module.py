# By Rohollah Moosavi Tayebi, email: rohollah.moosavi@uwaterloo.ca/moosavi.tayebi@gmail.com

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from xlwt import Workbook

import numpy as np
from data import load_all_images, save_data_list, get_data_loader, generate_fold, load_data_list
from model import get_densenet_model
import os
import sys
from collections import OrderedDict
device = "cuda" if torch.cuda.is_available() else "cpu"

def binary_acc(y_pred, y_test, device, threshold):
    if threshold == None:
        pred = torch.argmax(y_pred, dim=1)
    else:
        soft = F.softmax(y_pred)
        pred = torch.zeros(y_pred.shape[0]).to(device)
        for j in range(pred.shape[0]):
            if soft[j, 1] > threshold:
                pred[j] = 1
            else:
                pred[j] = 0

    correct_results_sum = (pred == y_test).sum().float()
    return correct_results_sum


def real_test(args):
    rgb_mean = (args.r_mean, args.g_mean, args.b_mean)
    rgb_std = (args.r_std, args.g_std, args.b_std)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(rgb_mean, rgb_std),
    ])

    print("loading data...")
    if args.load_cache:
        test_image_list, test_label_list, test_path_list = load_data_list(args.data_path + f'/data_{args.down_scale}.pt')
    else:
        test_image_list, test_label_list, test_path_list = load_all_images(args.data_path, args.labels_path, args.down_scale)
        save_data_list(test_image_list, test_label_list, test_path_list, args.data_path + f'/data_test_{args.down_scale}.pt')

    test_loader = get_data_loader(
        test_image_list,
        test_label_list,
        test_path_list,
        args.batch_size,
        shuffle=True,
        transform=transform_test
    )
    if args.loss_function == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    elif args.loss_function == 'BCELoss':
        criterion = nn.BCELoss()
    else:
        raise ValueError("Given loss function is not implemented yet!")

    best_model = torch.load(args.output_dir + '\BestModel.pt')

    torch.save(best_model.state_dict(), args.output_dir + '\BestModel_Dict.pt')

    model_name = args.model_name
    best_model = get_densenet_model(model_name)

    temp_dict = torch.load(args.output_dir + '\BestModel_Dict.pt')
    new_temp_dict = OrderedDict()

    for key, value in temp_dict.items():
        new_key = key[7:]
        new_temp_dict[new_key] = value

    best_model.load_state_dict(new_temp_dict)
    best_model1 = best_model.to(device)


    best_model1.eval()
    correct = 0
    z_list = []
    with torch.no_grad():
        for i, (x, y, z) in enumerate(test_loader):
            x = x.to(device)  
            y = y.to(device) 
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

            target = y 

            if i == 0:
                y_tot = target.cpu()
                pred_tot = pred.cpu()
            else:
                y_tot = torch.cat((y_tot.cpu(), target.cpu()), dim=0)
                pred_tot = torch.cat((pred_tot.cpu(), pred.cpu()), dim=0)

            loss = criterion(output, y).item()
            correct += binary_acc(output, y, device, args.threshold)
            if i == 0:
                test_losses = [loss]
            else:
                test_losses.append(loss)

            for p in z:
                z_list.append(p)

    accuracy = 100. * correct / len(test_loader.dataset.labels)

    FP = 0
    FN = 0
    TP = 0
    TN = 0

    TP_Path_List = []
    TN_Path_List = []
    FP_Path_List = []
    FN_Path_List = []

    for i in range(y_tot.shape[0]):
        if y_tot[i] == pred_tot[i]:
            if pred_tot[i] == 0:
                TN += 1
                TN_Path_List.append(z_list[i])
            else:
                TP += 1
                TP_Path_List.append(z_list[i])
        else:
            if pred_tot[i] == 0:
                FN += 1
                FN_Path_List.append(z_list[i])
            else:
                FP += 1
                FP_Path_List.append(z_list[i])

    if args.report_excel:
        wb = Workbook()

        sheetNumber0 = wb.add_sheet('Sheet 0')
        sheetNumber1 = wb.add_sheet('Sheet 1')
        sheetNumber2 = wb.add_sheet('Sheet 2')
        sheetNumber3 = wb.add_sheet('Sheet 3')
        sheetNumber4 = wb.add_sheet('Sheet 4')

        row = 0
        sheetNumber0.write(row, 0, 'TP List')
        sheetNumber0.write(row, 1, TP)
        row += 1
        for i in TP_Path_List:
            sheetNumber0.write(row, 0, i)
            row += 1

        row = 0
        sheetNumber1.write(row, 0, 'TN List')
        sheetNumber1.write(row, 1, TN)
        row += 1
        for i in TN_Path_List:
            sheetNumber1.write(row, 0, i)
            row += 1

        row = 0
        sheetNumber2.write(row, 0, 'FP List')
        sheetNumber2.write(row, 1, FP)
        row += 1
        for i in FP_Path_List:
            sheetNumber2.write(row, 0, i)
            row += 1

        row = 0
        sheetNumber3.write(row, 0, 'FN List')
        sheetNumber3.write(row, 1, FN)
        row += 1
        for i in FN_Path_List:
            sheetNumber3.write(row, 0, i)
            row += 1

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    specificity = TN /(TN + FP)
    NPV = TN / (TN + FN)

    if args.report_excel:
        sheetNumber4.write(2, 1, 'Accuracy')
        sheetNumber4.write(2, 2,int(accuracy.cpu().numpy()))
        sheetNumber4.write(3, 1, 'Precision (PPV)')
        sheetNumber4.write(3, 2, precision)
        sheetNumber4.write(4, 1, 'Recall (Sensitivity)')
        sheetNumber4.write(4, 2, recall)
        sheetNumber4.write(5, 1, 'Specificity')
        sheetNumber4.write(5, 2, specificity)
        sheetNumber4.write(6, 1, 'NPV')
        sheetNumber4.write(6, 2, NPV)
        wb.save(args.output_dir + f'\\test_result_fold_1.xls')

    print(f"Accuracy: {accuracy.cpu().numpy()}")
    print(f"Precision (PPV): {precision}")
    print(f"Recall (Sensitivity): {recall}")
    print(f"Specificity: {specificity}")
    print(f"NPV: {NPV}")