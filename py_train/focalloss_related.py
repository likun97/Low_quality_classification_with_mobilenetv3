
 
# https://github.com/yxdr/pytorch-multi-class-focal-loss/blob/master/FocalLoss.py
import torch
class FocalLoss:
    def __init__(self, alpha_t=None, gamma=0):
        """
        : alpha_t: A list of weights for each class
        : gamma:
        """
        self.alpha_t = torch.tensor(alpha_t) if alpha_t else None
        self.gamma = gamma



    def __call__(self, outputs, targets):
        if self.alpha_t is None and self.gamma == 0:
            focal_loss = torch.nn.functional.cross_entropy(outputs, targets)
            print('1111')

        elif self.alpha_t is not None and self.gamma == 0:
            if self.alpha_t.device != outputs.device:
                self.alpha_t = self.alpha_t.to(outputs)
            focal_loss = torch.nn.functional.cross_entropy(outputs, targets, weight=self.alpha_t)
            print('2222')

        elif self.alpha_t is None and self.gamma != 0:
            ce_loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='none')
            p_t = torch.exp(-ce_loss)
            focal_loss = ((1 - p_t) ** self.gamma * ce_loss).mean()
            print('3333')

        elif self.alpha_t is not None and self.gamma != 0:
            if self.alpha_t.device != outputs.device:
                self.alpha_t = self.alpha_t.to(outputs)
            ce_loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='none')
            p_t = torch.exp(-ce_loss)
            ce_loss = torch.nn.functional.cross_entropy(outputs, targets, weight=self.alpha_t, reduction='none')
            focal_loss = ((1 - p_t) ** self.gamma * ce_loss).mean()  # mean over the batch
            print('4444')

        return focal_loss


if __name__ == '__main__':
    outputs = torch.tensor([[2, 1.], [2.5, 1]], device='cuda')
    targets = torch.tensor([0, 1], device='cuda')

    print(torch.nn.functional.softmax(outputs, dim=1))

    fl= FocalLoss([0.5, 0.5], 2)

    print(fl(outputs, targets))

 
# ZENG   && https://www.jianshu.com/p/30043bcc90b6
class FocalLoss(nn.Module):
    def __init__(self, gamma = 2, eps = 1e-7):
        super(FocalLoss, self).__init__()
        self.eps    = eps
        self.gamma  = gamma
        self.ce    = nn.CrossEntropyLoss()



    def forward(self, input, target):

        logp = self.ce(input, target) 
        p    = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp

        return loss.mean()
 
# https://www.cnblogs.com/yumoye/p/11252909.html 
import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

def compute_class_weights(histogram):
    classWeights = np.ones(6, dtype=np.float32)
    normHist = histogram / np.sum(histogram)
    for i in range(6):
        classWeights[i] = 1 / (np.log(1.10 + normHist[i]))
    return classWeights
def focal_loss_my(input,target):
    '''
    :param input: shape [batch_size,num_classes,H,W]  
    :param target: shape [batch_size,H,W]
    :return:
    '''
    n, c, h, w = input.size()

    target = target.long()
    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.contiguous().view(-1)

    number_0 = torch.sum(target == 0).item()
    number_1 = torch.sum(target == 1).item()
    number_2 = torch.sum(target == 2).item()
    number_3 = torch.sum(target == 3).item()
    number_4 = torch.sum(target == 4).item()
    number_5 = torch.sum(target == 5).item()

    frequency = torch.tensor((number_0, number_1, number_2, number_3, number_4, number_5), dtype=torch.float32)
    frequency = frequency.numpy()
    classWeights = compute_class_weights(frequency)
    

    # weights=torch.from_numpy(classWeights).float().cuda()
    weights = torch.from_numpy(classWeights).float()
    focal_frequency = F.nll_loss(F.softmax(input, dim=1), target, reduction='none')
    

    focal_frequency += 1.0#shape  [num_samples]  1-P（gt_classes）

    focal_frequency = torch.pow(focal_frequency, 2)  # torch.Size([75])
    focal_frequency = focal_frequency.repeat(c, 1) 
    focal_frequency = focal_frequency.transpose(1, 0)
    loss = F.nll_loss(focal_frequency * (torch.log(F.softmax(input, dim=1))), target, weight=None,reduction='mean')
    return loss

def compute_class_weights(histogram):
    classWeights = np.ones(6, dtype=np.float32)
    normHist = histogram / np.sum(histogram)
    for i in range(6):
        classWeights[i] = 1 / (np.log(1.10 + normHist[i]))
    return classWeights
def focal_loss_zhihu(input, target):
    '''
    :param input:   https://zhuanlan.zhihu.com/p/28527749
    :param target:
    :return:
    '''
    n, c, h, w = input.size()

    target = target.long()
    inputs = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.contiguous().view(-1)

    N = inputs.size(0)
    C = inputs.size(1)

    number_0 = torch.sum(target == 0).item()
    number_1 = torch.sum(target == 1).item()
    number_2 = torch.sum(target == 2).item()
    number_3 = torch.sum(target == 3).item()
    number_4 = torch.sum(target == 4).item()
    number_5 = torch.sum(target == 5).item()

    frequency = torch.tensor((number_0, number_1, number_2, number_3, number_4, number_5), dtype=torch.float32)
    frequency = frequency.numpy()
    classWeights = compute_class_weights(frequency)

    weights = torch.from_numpy(classWeights).float()
    weights=weights[target.view(-1)] # important

    gamma = 2

    P = F.softmax(inputs, dim=1)#shape [num_samples,num_classes]

    class_mask = inputs.data.new(N, C).fill_(0)
    class_mask = Variable(class_mask)
    ids = target.view(-1, 1)
    class_mask.scatter_(1, ids.data, 1.)#shape [num_samples,num_classes]  one-hot encoding

    probs = (P * class_mask).sum(1).view(-1, 1)#shape [num_samples,]
    log_p = probs.log()

    print('in calculating batch_loss',weights.shape,probs.shape,log_p.shape)

    # batch_loss = -weights * (torch.pow((1 - probs), gamma)) * log_p
    batch_loss = -(torch.pow((1 - probs), gamma)) * log_p

    print(batch_loss.shape)

    loss = batch_loss.mean()
    return loss

if __name__=='__main__':
    pred=torch.rand((2,6,5,5))
    y=torch.from_numpy(np.random.randint(0,6,(2,5,5)))
    loss1=focal_loss_my(pred,y)
    loss2=focal_loss_zhihu(pred,y)

    print('loss1',loss1)
    print('loss2', loss2)
'''
in calculating batch_loss torch.Size([50]) torch.Size([50, 1]) torch.Size([50, 1])
torch.Size([50, 1])
loss1 tensor(1.3166)
loss2 tensor(1.3166)
'''
