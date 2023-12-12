# By Rohollah Moosavi Tayebi, email: rohollah.moosavi@uwaterloo.ca/moosavi.tayebi@gmail.com

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import xlrd
from xlwt import Workbook

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


def train(model, train_loader, epoch, criterion, optimizer, log_interval, device, scheduler=None):
    model.train()
    for batch_idx, (x, y, z) in enumerate(train_loader):
        if scheduler:
            scheduler.step()
        x = x.to(device)
        y = y.to(device)
        pred = model(x)

        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # For Showing logs:
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()), flush=True)


def test(args, model, test_loader, criterion, device, fold_num, report_excel, threshold=None):
    model.eval()
    correct = 0
    z_list = []
    with torch.no_grad():
        for i, (x, y, z) in enumerate(test_loader):
            x = x.to(device)  
            y = y.to(device)  
            output = model(x)
            if threshold == None:
                pred = torch.argmax(output, dim=1)
            else:
                soft = F.softmax(output)
                pred = torch.zeros(output.shape[0])
                for j in range(pred.shape[0]):
                    if soft[j, 1] > threshold:
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

            correct += binary_acc(output, y, device, threshold)
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

    if report_excel:
        wb = Workbook()

        sheetNumber0 = wb.add_sheet('Sheet 0')
        sheetNumber1 = wb.add_sheet('Sheet 1')
        sheetNumber2 = wb.add_sheet('Sheet 2')
        sheetNumber3 = wb.add_sheet('Sheet 3')

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

        wb.save(args.output_dir+ f'/test_result_fold_{fold_num}.xls')


    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    specificity = TN /(TN + FP)
    NPV = TN / (TN + FN)


    return accuracy.cpu().numpy(), precision, recall, specificity, NPV, np.array(test_losses)


def train_model(args, model, train_loader, validation_loader, n_epochs, criterion, optimizer, era, log_interval, device, threshold=None):
    best_precision = -1

    for epoch in range(1, n_epochs + 1):
        train(model, train_loader, epoch, criterion, optimizer, log_interval, device)
        accuracy, precision, recall, specificity, NPV, test_losses = test(args, model, validation_loader, criterion, device, 0, False, threshold=threshold)

        if precision >= best_precision:
            torch.save(model, args.output_dir + f'/BestModel_{era+1}.pt')

            print("best model saved!")
            best_acc = accuracy
            best_precision = precision
            best_recall = recall
            best_specificity = specificity
            best_NPV = NPV

        print("*" * 50)
        print(f"\t End of epoch {epoch}/{n_epochs} from era {era+1}")
        print('-' * 50)
        print(f"\t accuracy: {accuracy}")
        print(f"\t precision: {precision}")
        print(f"\t recall: {recall}")
        print(f"\t specificity: {specificity}")
        print(f"\t NPV: {NPV}")
        print(f"\t test average loss: {np.mean(test_losses)}")
        print("*" * 50, flush=True)

    return best_acc.item(), best_precision, best_recall, best_specificity, best_NPV
