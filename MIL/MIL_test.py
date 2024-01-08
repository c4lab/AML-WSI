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
    parser.add_argument('--lib', type=str, default='filelist', help='path to data file')
    parser.add_argument('--patches_path', type=str, default='', help='path to patches')
    parser.add_argument('--output', type=str, default='.', help='name of output directory')
    parser.add_argument('--model', type=str, default='', help='path to pretrained model')
    parser.add_argument('--CNN', type=str, default='resnet', help='CNN model')
    parser.add_argument('--batch_size', type=int, default=64, help='how many images to sample per slide (default: 64)')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
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

class MILdataset(data.Dataset):
    def __init__(self, libraryfile='', transform=None, model_name="resnet34",slide_path=""):
        if model_name in ["densenet121","densenet161","densenet201","VGG19"]:
            self.resize = False
        else:
            self.resize = True 
        lib = torch.load(libraryfile)
        print('loading', libraryfile)
        #Flatten grid
        grid = []
        slideIDX = []
        for i,g in enumerate(lib['grid']):
            grid.extend(g)
            slideIDX.extend([i]*len(g))
            print('Slide {} has {} tiles'.format(i+1,len(g)))

        print('Number of tiles: {}'.format(len(grid)))
        self.slidenames = lib['slides']
        self.targets = lib['targets']
        self.grid = grid
        self.slideIDX = slideIDX
        self.transform = transform
        self.mode = None
        self.mult = lib['mult']
        self.size = int(np.round(224*lib['mult']))
        self.level = lib['level']
        self.slide_path = slide_path
    def setmode(self,mode):
        self.mode = mode
    def maketraindata(self, idxs):
        self.t_data = [(self.slideIDX[x],self.grid[x],self.targets[self.slideIDX[x]]) for x in idxs]
    def shuffletraindata(self):
        self.t_data = random.sample(self.t_data, len(self.t_data))
    def __getitem__(self,index):
        if self.mode == 1:
            slide_num = self.slidenames[self.slideIDX[index]].split("_")[0]
            slide_path = f"{self.slide_path}{slide_num}/"
            patch_path = slide_path+ self.grid[index]
            img =Image.open(patch_path).convert('RGB')
            # if self.mult != 1:
            if self.resize:
                img = img.resize((224,224),Image.BILINEAR)
            if self.transform is not None:
                img = self.transform(img)
            return img
        elif self.mode == 2:
            slideIDX, patch, target = self.t_data[index]
            slide_num = self.slidenames[slideIDX].split("_")[0]
            slide_path = f"{self.slide_path}{slide_num}/"
            img = Image.open(slide_path+patch).convert('RGB')
            if self.resize:
                img = img.resize((224,224),Image.BILINEAR)
            if self.transform is not None:
                img = self.transform(img)
            return img, target
    def __len__(self):
        if self.mode == 1:
            return len(self.grid)
        elif self.mode == 2:
            return len(self.t_data)

def inference(run, loader, model):
    model.eval()
    probs = torch.FloatTensor(len(loader.dataset))
    with torch.no_grad():
        for i, input in enumerate(loader):
            print('Inference\tEpoch: [{}/{}]\tBatch: [{}/{}]'.format(run+1, args.nepochs, i+1, len(loader)))
            input = input.cuda()
            output = F.softmax(model(input), dim=1)
            probs[i*args.batch_size:i*args.batch_size+input.size(0)] = output.detach()[:,1].clone()
    return probs.cpu().numpy()

def group_max(groups, data, nmax):
    out = np.empty(nmax)
    out[:] = np.nan
    order = np.lexsort((data, groups))
    groups = groups[order]
    data = data[order]
    index = np.empty(len(groups), 'bool')
    index[-1] = True
    index[:-1] = groups[1:] != groups[:-1]
    out[groups[index]] = data[index]
    return out

def main(args):
    model = get_pretrained_model(args.CNN)
    ch = torch.load(args.model)
    model.load_state_dict(ch['state_dict'])
    model.cuda()
    cudnn.benchmark = True

    #normalization
    normalize = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.1,0.1,0.1])
    trans = transforms.Compose([transforms.ToTensor(), normalize])

    #load data
    dset = MILdataset(args.lib, trans)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    dset.setmode(1)
    probs = inference(loader, model)
    maxs = group_max(np.array(dset.slideIDX), probs, len(dset.targets))

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    fp = open(os.path.join(args.output, 'predictions.csv'), 'w')
    fp.write('file,target,prediction,probability\n')
    for name, target, prob in zip(dset.slidenames, dset.targets, maxs):
        fp.write('{},{},{},{}\n'.format(name, target, int(prob>=0.5), prob))
    fp.close()

if __name__ == "__main__":
    start_time = time.time()
    args = get_args()
    # run the main function
    main(args)
    exec_time = time.time() - start_time
    print("time: {:02d}m{:02d}s".format(int(exec_time // 60), int(exec_time % 60)))