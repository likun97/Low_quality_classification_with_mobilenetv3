import json
import torch.utils.data as Data
import os
from PIL import Image
import random
import numpy as np
import torch
from torchvision.transforms import transforms
# from utils import load_images
# from utils import data_augmentation
# import glob
from PIL import ImageFilter


def read_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


# class TheDataset(Data.Dataset):
#     def __init__(self, data_root, trainJason):
#         super(TheDataset, self).__init__()
#         self.root = data_root

#         data_dict = read_json(trainJason)            
#         self.img_path_list = list(data_dict.keys())
#         self.gt_path_list = list(data_dict.values())


#     def __getitem__(self, item):

#         rand_mode = random.randint(0, 7)
#         img_path = os.path.join(self.root, self.img_path_list[item])
#         gt = self.gt_path_list[item]
#         img = load_images(img_path)
#         img = data_augmentation(img, rand_mode)
#         img = torch.from_numpy(img.copy()).permute(2, 0, 1)
#         return img, gt

#     def __len__(self):
#         return len(self.img_path_list)

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class TheDataset(Data.Dataset):
    def __init__(self, data_root):

        self.image_root = data_root

        total_pairs = []
        lines = open('/search/gpu4/.../train.txt').read().split('\n')
        for line in lines:
            line = line.strip()
            # print(line)
            if line and os.path.exists(line.split('---->')[0]):
                tfs, s_idx = line.split('---->')     
                s_idx = int(s_idx)
                total_pairs.append((tfs, s_idx))
        # print(total_pairs)
        self.pairs = total_pairs        
        # self.img_size = 224
        self.transform = transforms.Compose([
            transforms.Scale((224,224)),
            transforms.RandomCrop(224, padding=4),
            # transforms.RandomApply([
            #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            # ], p=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __getitem__(self, index):
        image_file, s_idx = self.pairs[index]
        # print(image_file, s_idx)
        # image_file = os.path.join(self.image_root, image_file)
        image = Image.open(image_file).convert('RGB')
        image = self.transform(image)
        return image, s_idx

    def __len__(self):
        return len(self.pairs)


class TheDatasettest(Data.Dataset):
    def __init__(self, data_root):

        self.image_root = data_root

        total_pairs = []
        lines = open('/search/gpu4/.../test.txt').read().split('\n')
        for line in lines:
            line = line.strip()
            if line:
                tfs, s_idx = line.split('---->')    
                s_idx = int(s_idx)
                total_pairs.append((tfs, s_idx))
        # print(total_pairs)
        self.pairs = total_pairs        
        # self.img_size = 224
        self.transform = transforms.Compose([
            transforms.Scale((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __getitem__(self, index):
        image_file, s_idx = self.pairs[index]
        # print(image_file, s_idx)
        # image_file = os.path.join(self.image_root, image_file)
        image = Image.open(image_file).convert('RGB')
        image = self.transform(image)
        return image, s_idx

    def __len__(self):
        return len(self.pairs)


if __name__ == '__main__':
    data = TheDataset(data_root='/search/gpu4.../train')
    dataloader = torch.utils.data.DataLoader(data, batch_size=16, shuffle=True,
                                             num_workers=0, pin_memory=False)
    for (img, gt) in dataloader:
        print(img.size())
