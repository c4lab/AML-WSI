import os
import time
import numpy as np
import argparse
import random
import PIL.Image as Image
import torch
import gc
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_lib', type=str, default='', help='path to train MIL library binary')
    parser.add_argument('--val_lib', type=str, default='', help='path to validation MIL library binary. If present.')
    parser.add_argument('--slide_path', type=str, default='', help='path to slide')
    parser.add_argument('--output', type=str, default='.', help='name of output file')
    parser.add_argument('--CNN', type=str, default='resnet34', help='CNN model')
    parser.add_argument('--batch_size', type=int, default=512, help='mini-batch size (default: 512)')
    parser.add_argument('--nepochs', type=int, default=100, help='number of epochs (default: 100)')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--test_every', default=1, type=int, help='test on val every (default: 1)')
    parser.add_argument('--weights', default=0.5, type=float, help='unbalanced positive class weight (default: 0.5, balanced classes)')
    parser.add_argument('--k', default=1, type=int, help='top k tiles are assumed to be of the same class as the slide (default: 1, standard MIL)')
    args = parser.parse_args()
    return args
def get_pretrained_model(model_name):
    if (model_name == "densenet121"):
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        num_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(num_features, 2)
    elif(model_name == "densenet161"):
        model = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
        num_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(num_features, 2)
    elif(model_name == "densenet201"):
        model = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT)
        num_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(num_features, 2)
    elif(model_name == "efficientnetV2"):
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        num_features = model.classifier[1].in_features
        model._fc = torch.nn.Linear(num_features, 2)
    elif(model_name == "VGG19"):
        model = models.vgg19_bn(weights=models.VGG19_BN_Weights.DEFAULT)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
    elif(model_name == "resnext"):
        model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif(model_name == "alexnet"):
        model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
    elif(model_name == "resnet50"):
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif(model_name == "resnet101"):
        model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif(model_name == "resnet152"):
        model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 2)    
    else:
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 2)

    return model

def main(args):
    best_acc = 0
    model = get_pretrained_model(args.CNN)
    model.cuda()

    if args.weights==0.5:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        w = torch.Tensor([1-args.weights,args.weights])
        criterion = nn.CrossEntropyLoss(w).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    cudnn.benchmark = True

    #normalization
    normalize = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.1,0.1,0.1])
    trans = transforms.Compose([transforms.ToTensor(), normalize])

    #load data
    train_dset = MILdataset(args.train_lib, trans)
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)
    if args.val_lib:
        val_dset = MILdataset(args.val_lib, trans)
        val_loader = torch.utils.data.DataLoader(
            val_dset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=False)

    #open output file
    fconv = open(os.path.join(args.output,'convergence.csv'), 'w')
    fconv.write('epoch,metric,value\n')
    fconv.close()

    #loop throuh epochs
    for epoch in range(args.nepochs):
        train_dset.setmode(1)
        probs = inference(epoch, train_loader, model)
        topk = group_argtopk(np.array(train_dset.slideIDX), probs, args.k)
        train_dset.maketraindata(topk)
        train_dset.shuffletraindata()
        train_dset.setmode(2)
        loss = train(epoch, train_loader, model, criterion, optimizer)
        print('Training\tEpoch: [{}/{}]\tLoss: {}'.format(epoch+1, args.nepochs, loss))
        fconv = open(os.path.join(args.output, 'convergence.csv'), 'a')
        fconv.write('{},loss,{}\n'.format(epoch+1,loss))
        fconv.close()
        # clearing the occupied cuda memory 
        gc.collect()
        torch.cuda.empty_cache()
        #Validation
        if args.val_lib and (epoch+1) % args.test_every == 0:
            val_dset.setmode(1)
            probs = inference(epoch, val_loader, model)
            maxs = group_max(np.array(val_dset.slideIDX), probs, len(val_dset.targets))
            pred = [1 if x >= 0.5 else 0 for x in maxs]
            err,fpr,fnr = calc_err(pred, val_dset.targets)
            print('Validation\tEpoch: [{}/{}]\tError: {}\tFPR: {}\tFNR: {}'.format(epoch+1, args.nepochs, err, fpr, fnr))
            fconv = open(os.path.join(args.output, 'convergence.csv'), 'a')
            fconv.write('{},error,{}\n'.format(epoch+1, err))
            fconv.write('{},fpr,{}\n'.format(epoch+1, fpr))
            fconv.write('{},fnr,{}\n'.format(epoch+1, fnr))
            fconv.close()
            #Save best model
            err = (fpr+fnr)/2.
            if 1-err >= best_acc:
                best_acc = 1-err
                obj = {
                    'epoch': epoch+1,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict()
                }
                torch.save(obj, os.path.join(args.output,'checkpoint_best.pth'))


if __name__ == "__main__":
    start_time = time.time()
    args = get_args()
    # run the main function
    main(args)
    exec_time = time.time() - start_time
    print("time: {:02d}m{:02d}s".format(int(exec_time // 60), int(exec_time % 60)))