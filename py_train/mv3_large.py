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
from my_pooling import my_MaxPool2d,my_AvgPool2d
import torchvision.transforms as transforms
from dataset import TheDataset,TheDatasettest
import math
from torchsampler import ImbalancedDatasetSampler
import torch.utils.model_zoo as model_zoo
from torchvision import models
from PIL import ImageFile
from visual import draw_CAM
from thop import profile
from torchstat import stat
from PIL import ImageFilter
from torch.utils.data import WeightedRandomSampler
import collections
import pandas as pd
ImageFile.LOAD_TRUNCATED_IMAGES = True


logger = logging.getLogger('MC_VGG_224')
 

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

 

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
 

lr = 0.001
nb_epoch = 300
 
#Data
print('==> Preparing data..') 

data_root='/search/odin/xxx/'

traindir = os.path.join(data_root, 'train')
valdir = os.path.join(data_root, 'test')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.7),    # not strengthened
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))
train_sampler = None


## 
idx = 0
train_classes = [label for _, label in train_dataset.imgs]
# print(train_classes)
class_count =  collections.Counter(train_classes)
class_weights = torch.Tensor([len(train_classes)/c for c in pd.Series(class_count).sort_index().values])
sample_weights = [0] * len(train_dataset)

for image, label in train_dataset.imgs:
    class_weight = class_weights[label]
    sample_weights[idx] = class_weight
    idx += 1


# print(sample_weights)
train_sampler = WeightedRandomSampler(weights=sample_weights,  num_samples = len(train_dataset), replacement=True)

# train_sampler=ImbalancedDatasetSampler(train_dataset)


trainloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=200, shuffle=(train_sampler is None),        #  1024
    num_workers=24, pin_memory=True, sampler=train_sampler)
 

# 1 train_dataset = datasets.ImageFolder() 
#                                           
# 2 train_sampler = ImbalancedDatasetSampler(train_dataset)   
#                                                           
# 3 trainloader   = torch.utils.data.DataLoader( train_dataset,

# 4 for batch_idx, (inputs, targets) in enumerate(trainloader): 

testloader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Scale((300,300)),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=60, shuffle=False,                                  
    num_workers=24, pin_memory=True)
 
print('==> Building model..') 

# criterion = nn.CrossEntropyLoss().cuda() #  loss with softmax   
# no use of softmax in the end https://zhuanlan.zhihu.com/p/159477597  

print('==> Building model..')

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, eps=1e-7):
        
        super(FocalLoss, self).__init__()
        self.eps    = eps 
        self.gamma  = gamma
        self.ce     = nn.CrossEntropyLoss() 

    def forward(self, input, target):
        logp = self.ce(input, target)
        p    = torch.exp(-logp)
        loss = (1-p)**self.gamma * logp

        return loss.mean()

criterion = FocalLoss().cuda()  
 
__all__ = ['MobileNetV3', 'mobilenetv3']


def conv_bn(inp, oup, stride, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 3, stride, 1, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)         ## https://blog.csdn.net/qq_35037684/article/details/109050296
    )
 
def conv_1x1_bn(inp, oup, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 1, 1, 0, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.
 
class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)     
 
class MobileBottleneck(nn.Module):

    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE'):
        super(MobileBottleneck, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup    
        
        conv_layer = nn.Conv2d
        norm_layer = nn.BatchNorm2d
        if nl == 'RE':
            nlin_layer = nn.ReLU # or ReLU6
        elif nl == 'HS':
            nlin_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        self.conv = nn.Sequential(
            # pw  
            conv_layer(inp, exp, 1, 1, 0, bias=False),        
            norm_layer(exp),
            nlin_layer(inplace=True),
            # dw                                    
            conv_layer(exp, exp, kernel, stride, padding, groups=exp, bias=False),
            norm_layer(exp),
            SELayer(exp),
            nlin_layer(inplace=True),
            # pw-linear
            conv_layer(exp, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x) 

class MobileNetV3(nn.Module):
    def __init__(self, n_class=9, input_size=224, dropout=0.5, mode='large', width_mult=2.0):     

        super(MobileNetV3, self).__init__()

        if mode == 'large':
            # refer to Table 1 in paper
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  False, 'RE', 1],
                [3, 64,  24,  False, 'RE', 2],
                [3, 72,  24,  False, 'RE', 1],
                [5, 72,  40,  True,  'RE', 2],
                [5, 120, 40,  True,  'RE', 1],
                [5, 120, 40,  True,  'RE', 1],
                [3, 240, 80,  False, 'HS', 2],
                [3, 200, 80,  False, 'HS', 1],
                [3, 184, 80,  False, 'HS', 1],
                [3, 184, 80,  False, 'HS', 1],
                [3, 480, 112, True,  'HS', 1],
                [3, 672, 112, True,  'HS', 1],
                [5, 672, 160, True,  'HS', 2],
                [5, 960, 160, True,  'HS', 1],
                [5, 960, 160, True,  'HS', 1],
            ]
        elif mode == 'small':
            # refer to Table 2 in paper
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  True,  'RE', 2],
                [3, 72,  24,  False, 'RE', 2],
                [3, 88,  24,  False, 'RE', 1],
                [5, 96,  40,  True,  'HS', 2],
                [5, 240, 40,  True,  'HS', 1],
                [5, 240, 40,  True,  'HS', 1],
                [5, 120, 48,  True,  'HS', 1],
                [5, 144, 48,  True,  'HS', 1],
                [5, 288, 96,  True,  'HS', 2],
                [5, 576, 96,  True,  'HS', 1],
                [5, 576, 96,  True,  'HS', 1],
            ]
        else:
            raise NotImplementedError

        input_channel = 16
        last_channel  = 1280

        # building first layer
        assert input_size % 32 == 0                                                             
        last_channel    = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features   = [conv_bn(3, input_channel, 2, nlin_layer=Hswish)]         
        self.classifier = []

 

        # building mobile blocks   
        for k, exp, c, se, nl, s in mobile_setting:              # [3, 16,  16,  False, 'RE', 1]

            output_channel = make_divisible(c * width_mult)     
            exp_channel    = make_divisible(exp * width_mult)
            self.features.append(MobileBottleneck(input_channel, output_channel, k, s, exp_channel, se, nl))
            input_channel  = output_channel




        # building last several layers  
        if mode == 'large':
            last_conv = make_divisible(960 * width_mult)
            self.features.append(conv_1x1_bn(input_channel, last_conv, nlin_layer=Hswish))
            self.features.append(nn.AdaptiveAvgPool2d(1))
            self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
            self.features.append(Hswish(inplace=True))
        elif mode == 'small':
            last_conv = make_divisible(576 * width_mult)
            self.features.append(conv_1x1_bn(input_channel, last_conv, nlin_layer=Hswish))
            # self.features.append(SEModule(last_conv))  # refer to paper Table2, but I think this is a mistake
            self.features.append(nn.AdaptiveAvgPool2d(1))
            self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
            self.features.append(Hswish(inplace=True))
        else:
            raise NotImplementedError
 
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)
  
        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),    # refer to paper section 6
            nn.Linear(last_channel, n_class),
        )

        self._initialize_weights()
 
    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


def mobilenetv3(pretrained=False, **kwargs):
    model = MobileNetV3(**kwargs)
    if pretrained:
        state_dict = torch.load('mobilenetv3_small_67.4.pth.tar')
        model.load_state_dict(state_dict, strict=True)
        # raise NotImplementedError
    return model
 

use_cuda = torch.cuda.is_available() 
net = mobilenetv3() 


if use_cuda:
    net.classifier.cuda()
    net.features.cuda()

    net.classifier = torch.nn.DataParallel(net.classifier)
    net.features = torch.nn.DataParallel(net.features)

    cudnn.benchmark = True  
    
def train(epoch,net, args, trainloader,optimizer):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    idx = 0
    
    # with open('error.txt', 'w') as f:
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        idx = batch_idx

        inputs, targets = inputs.cuda(), targets.cuda()
        # inputs, targets = inputs, targets
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)

        #out, ce_loss, MC_loss = net(inputs, targets)

        out = net(inputs)

        ce_loss = criterion(out, targets)
 
        loss = ce_loss # + args["alpha_1"] * MC_loss[0] +   args["beta_1"]  * MC_loss[1] 
        # print("loss======",loss)
        loss.backward()
        optimizer.step()
 
        train_loss += loss.item()

        _, predicted = torch.max(out.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()
        # errors = predicted.ne(targets.data)
        # index = 0
        # for error in errors:
        #     if error.cpu().item()==0:
        #         index += 1
        #         continue
        #     f.write(img_file[index] + '---->' + str(predicted[index].cpu().item()) + '   lable:' + str(targets[index].cpu().item()) + '\n')
        #     index += 1

        if batch_idx % 10 == 0:
            print('epoch[%d]---Train, train_acc = %.4f,train_loss = %.4f' % (epoch, 100.*correct/total,train_loss/(idx+1)))
 
    train_acc = 100.*correct/total
    train_loss = train_loss/(idx+1)
    logging.info('Iteration %d, train_acc = %.5f,train_loss = %.6f' % (epoch, train_acc, train_loss))

    return train_acc, train_loss
 
def test(epoch,net,testloader,optimizer):

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    idx = 0
    
    # with open('output.txt', 'w')  as outf:
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # name = inputs
        with torch.no_grad():
            idx = batch_idx
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
 
            out = net(inputs)

            ce_loss = criterion(out, targets)
 
            test_loss += ce_loss.item()
            _, predicted = torch.max(out.data, 1)
            # outf.write(str(name[0]) + '---->' + str(predicted.cpu().item())+'\n')
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().item()

        if batch_idx % 10 == 0:
            print('epoch[%d]---Test, test_acc = %.4f,test_loss = %.4f' % (epoch, 100.*correct/total,test_loss/(idx+1)))


    test_acc = 100.*correct/total
    test_loss = test_loss/(idx+1)
    logging.info('test, test_acc = %.4f,test_loss = %.4f' % (test_acc,test_loss))

 
    return test_acc
 


optimizer = optim.SGD([
                        {'params': net.classifier.parameters(), 'lr': 0.001},
                        {'params': net.features.parameters(),   'lr': 0.001},
                        
                     ], 
                      momentum=0.9, weight_decay=5e-4) 

def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MC2_AutoML Example')

    parser.add_argument('--alpha_1', type=float, default=1.5, metavar='ALPHA', help='alpha_1 value (default: 2.0)')
    parser.add_argument('--beta_1' , type=float, default=20.0, metavar='BETA',  help='beta_1 value (default: 20.0)')
    parser.add_argument('--resume' , default='/search/odin/xxx/xxx/model/model_large_2.pth', type=str, metavar='PATH',
                                      help='path to latest checkpoint (default: none)')

    args, _ = parser.parse_known_args()
    return args 

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    print('---save once now')
    torch.save(state, filename )  

if __name__ == '__main__':

    #x = torch.randn(input_size)
    #out = net(x)


    try:
        args = vars(get_params())
        print(args)
        # main(params)
        max_val_acc = 0

        if args["resume"]:
            if os.path.isfile(args["resume"]):
                print("=> loading checkpoint '{}'".format(args["resume"]))
                checkpoint = torch.load(args["resume"])
                max_val_acc = checkpoint['best_prec1']          # checkpoint['best_prec1']
                net.load_state_dict(checkpoint['state_dict_'])  # checkpoint['state_dict_']   
            else:
                print("=> no checkpoint found at '{}'".format(args["resume"]))

 
        for epoch in range(1, nb_epoch+1):
            if epoch ==70:
                lr = 0.0003
            if epoch ==100:
                lr = 0.0001
            optimizer.param_groups[0]['lr'] = lr
            optimizer.param_groups[1]['lr'] = lr 

            train(epoch, net, args,trainloader,optimizer) 
            test_acc = test(epoch, net,testloader,optimizer)

            if test_acc >max_val_acc:
                max_val_acc = test_acc
                save_checkpoint(
                    {'state_dict_': net.state_dict(),  'best_prec1': max_val_acc, }, 
                    filename=os.path.join('/search/odin/xxx/xxx/model', 'model_large_3.pth')
                    )

            print("max_val_acc", max_val_acc)
            print("test_acc", test_acc)
             

        # test_acc = test(1, net,testloader,optimizer)
    except Exception as exception:
        logger.exception(exception)
        raise
  
# model_large_1      drop_out0.8   ImbalancedDatasetSampler   loss = nn.CrossEntropyLoss()     width  1      
# model_large_2      drop_out0.5   WeightedRandomSampler      loss = FocalLoss().cuda()        width  1       
# model_large_3      drop_out0.5   WeightedRandomSampler      loss = FocalLoss().cuda()        width  2.
# loss = FocalLoss().cuda() 
# https://github.com/yxdr/pytorch-multi-class-focal-loss/blob/master/FocalLoss.py 



 


 
