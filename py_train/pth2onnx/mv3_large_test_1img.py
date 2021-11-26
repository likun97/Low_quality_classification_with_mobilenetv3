'''PyTorch CUB-200-2011 Training with VGG16 (TRAINED FROM SCRATCH).'''
from __future__ import print_function
import os
# import nni
import time
import torch
import logging
import argparse
import torchvision
import random
from PIL import Image
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
from dataset import TheDataset,TheDatasettest
import math
from torchsampler import ImbalancedDatasetSampler
import torch.utils.model_zoo as model_zoo
from torchvision import models
from PIL import ImageFile
# from visual import draw_CAM
from thop import profile
from torchstat import stat
from PIL import ImageFilter
from torch.utils.data import WeightedRandomSampler
import collections
import pandas as pd
ImageFile.LOAD_TRUNCATED_IMAGES = True


import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))
# import onnx 
import torch
from mv3_resnet50_large_2 import mobilenetv3, MobileNetV3
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
net = mobilenetv3().cuda() 
         
# net = MobileNetV3()###.cuda()  
 

logger = logging.getLogger('MC_VGG_224')
 
data_root='/search/odin/likun'
 
valdir = os.path.join(data_root, 'test_6')
 
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

testloader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Scale((300,300)),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=10, shuffle=False,                                 
    num_workers=24, pin_memory=True)
 

use_cuda = torch.cuda.is_available()

if use_cuda:
    net.classifier.cuda()
    net.features.cuda()

    net.classifier = torch.nn.DataParallel(net.classifier)
    net.features = torch.nn.DataParallel(net.features)

    cudnn.benchmark = True


criterion = nn.CrossEntropyLoss().cuda() 

def test(epoch,net,testloader):

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    idx = 0
    topk_ids   = [] 
    with open('output.txt', 'w')  as outf:

        for batch_idx, (inputs, targets) in enumerate(testloader):
 
            name = inputs

            with torch.no_grad():
                idx = batch_idx
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = Variable(inputs), Variable(targets)
 
                out = net(inputs)

                ce_loss = criterion(out, targets)
    
                test_loss += ce_loss.item()

                _, predicted = torch.max(out.data, 1)

                print('---', out)
                print('---输出类别', predicted)
                print('---真实类别', targets.data)
 
                topk_ids.append(predicted.cpu().numpy())   
  
                # outf.write(str(name[0]) + '---->' + str(predicted.cpu().item())+'\n')

                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum().item()

            if batch_idx % 10 == 0:
                print('---Test, test_acc = %.4f,test_loss = %.4f' % (100.*correct/total,test_loss/(idx+1)))

    topk_ids = np.concatenate(topk_ids, axis=0)

    test_acc = 100.*correct/total
    test_loss = test_loss/(idx+1)
    logging.info('test, test_acc = %.4f,test_loss = %.4f' % (test_acc,test_loss))
                
 
                
    with open(os.path.join('/search/odin/likun/zyresnet/test_large_v3_16_width111.csv'), 'w') as out_file:
        for label in  topk_ids:
            label = label.tolist()
            # print('---')
            # print(type(label))
            # print(label)
            # out_file.write('{0}\n'.format([ str(v) for v in label]))    
            out_file.write('{:01d}\n'.format(label))
            
    return test_acc
 

def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MC2_AutoML Example') 

    parser.add_argument('--resume' , default='/search/odin/likun/zyresnet/A_onnx/model_large_2_transfer.pth', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

    args, _ = parser.parse_known_args()
    return args
  
if __name__ == '__main__': 
    try:
        args = vars(get_params())
        print(args)
        # main(params)
        max_val_acc = 0
 
        if args["resume"]:
            if os.path.isfile(args["resume"]):
                print("=> loading checkpoint '{}'".format(args["resume"]))
                checkpoint = torch.load(args["resume"])

                # # max_val_acc = checkpoint['best_prec1']
                # for k, v in checkpoint['state_dict_'].items():
                #     print(k)

                # net.load_state_dict(checkpoint['state_dict_'], True)                
                net.load_state_dict(checkpoint['state_dict_'], False)   
 
                #  多机的模型单机测试  加载模型问题  直接加载 True 不行  多了关键字   False 能加载但其实是虚的  
                #  这里就好像 True  False  都能加载对模型    还是因为  81行  False的时候  其实用的是多机测试加载    
            else:
                print("=> no checkpoint found at '{}'".format(args["resume"]))
 

        test_acc = test(1, net,testloader)


    except Exception as exception:
        logger.exception(exception)
        raise



