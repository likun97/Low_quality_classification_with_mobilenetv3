
import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))
import onnx 
import torch
from mv3_resnet50_large_2 import mobilenetv3, MobileNetV3
import cv2
import onnxruntime
import numpy as np 


session = onnxruntime.InferenceSession("./model_large_2_v2_softmax_dim1_V2_dropout2_300.onnx") #  session 

path = '/search/odin/likun/test_6/8'

file_list = os.listdir(path)   
print(file_list)

for data_root in file_list:
    
    # print(data_root)
    img       = torch.from_numpy(cv2.cvtColor(cv2.resize(cv2.imread(path+'/'+data_root), (300, 300)), cv2.COLOR_BGR2RGB)[np.newaxis, :, :, :].astype(np.float32))
    img       = img.permute(0, 3, 1, 2) / 255.
    img[:, 0, :, :] = (img[:, 0, :, :] - 0.485)/0.229
    img[:, 1, :, :] = (img[:, 1, :, :] - 0.456)/0.224
    img[:, 2, :, :] = (img[:, 2, :, :] - 0.406)/0.225
     
    out_r = session.run(None, {"input": img.cpu().numpy()})                                  

    

    # print(out_r)  

    out = np.array(out_r) 
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=5)
    print(np.squeeze(out,axis = None)) 
    print("--------------------------------")

     
 
