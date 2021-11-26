# import sys
# import os 
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))
# import onnx 
import torch
# from mv3_resnet50_large_2 import mobilenetv3
  
# ############################ 
# model = mobilenetv3().cuda() 



# checkpoint = torch.load("/search/odin/likun/zyresnet/A_onnx/model_large_2_transfer.pth")
# model.load_state_dict(checkpoint['state_dict_'],False)      # 预训练模型加载     # net.load_state_dict(checkpoint['state_dict_'])


# export_onnx_file = "model_large_2.onnx"

# x=torch.onnx.export(model,                                  # 待转换的网络模型和参数
#                 torch.randn(1, 3, 300, 300, device='cuda'), # 虚拟的输入，用于确定输入尺寸和推理计算图每个节点的尺寸
#                 export_onnx_file,                           # 输出文件的名称
#                 verbose=False,                              # 是否以字符串的形式显示计算图
#                 input_names=["input"],                      # 输入节点的名称，
#                 output_names=["output"],                    # 输出节点的名称
#                 opset_version=10,                           # onnx 支持采用的operator set
#                 do_constant_folding=True,                   # 是否压缩常量
#                 dynamic_axes={"input":{0: "batch_size"}, "output":{0: "batch_size"},} 
#                 )                                           # #设置动态维度，此处指明input节点的第0维度可变，命名为batch_size，若设置固定的batch_size，则不需要设置dynamic_axes参数。
   
# net = onnx.load("model_large_2.onnx")    # 加载onnx 计算图
# onnx.checker.check_model(net)            # 检查文件模型是否正确

# import onnxruntime
# import numpy as np

# session = onnxruntime.InferenceSession("./model_large_2.onnx") # 创建一个运行session，类似于tensorflow
# out_r = session.run(None, {"input": np.random.rand(1, 3, 300, 300).astype('float32')})  # 模型运行，注意这里的输入必须是numpy类型
# print(len(out_r))
# print(out_r[0].shape) 
################################################################# 

# V2版本    model_large_2_v2_softmax_dim1_V2 把   mv3_resnet50_large_2.py  删除了一些

# import sys
# import os 
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))
# import onnx 
# import torch
# from mv3_resnet50_large_2 import mobilenetv3, MobileNetV3
# import cv2
# import onnxruntime
# import numpy as np
  
# model = mobilenetv3().cuda().eval()    # model = MobileNetV3.cuda
 
# checkpoint = torch.load("/search/odin/likun/zyresnet/A_onnx/model_large_2_transfer.pth")
 
# model.load_state_dict(checkpoint['state_dict_'], False)     
# export_onnx_file = "model_large_2_v2_softmax_dim1_V2_dropout1.onnx"
 
# x=torch.onnx.export(model,                                 
#                 torch.randn(1, 3, 300, 300, device='cuda'), 
#                 export_onnx_file,                          
#                 verbose=False,                              
#                 input_names=["input"],                    
#                 output_names=["output"],                   
#                 opset_version=10,                         
#                 do_constant_folding=True,                 
#                 # dynamic_axes={"input":{0: "batch_size"}, "output":{0: "batch_size"},} 
#                 )                                        

# net = onnx.load("model_large_2_v2_softmax_dim1_V2_dropout1.onnx")   
# onnx.checker.check_model(net)            

# import onnxruntime
# import numpy as np

# session = onnxruntime.InferenceSession("./model_large_2_v2_softmax_dim1_V2.onnx") 
# out_r = session.run(None, {"input": np.random.rand(1, 3, 300, 300).astype('float32')}) 
# print(len(out_r))
# print(out_r[0].shape) 
################################################################# 

# # V2      yangxiaohui ----- help 

# import sys
# import os 
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))
# import onnx 
# import torch
# from mv3_resnet50_large_2 import mobilenetv3, MobileNetV3
# import cv2
# import onnxruntime
# import numpy as np 
# data_root ='/search/odin/likun/test_6/8pad_imgs/1005.jpg_02.jpg'
# img       = torch.from_numpy(cv2.resize(cv2.imread(data_root), (300, 300))[np.newaxis, :, :, :].astype(np.float32))
# img       = img.permute(0, 3, 1, 2).cuda() / 255.


# # model   = MobileNetV3.cuda
# model     = mobilenetv3().cuda().eval()    ####  .eval() .eval()  
 
# checkpoint = torch.load("/search/odin/likun/zyresnet/A_onnx/model_large_2_transfer.pth")

# ### ---0
# # model.load_state_dict(checkpoint['state_dict_'], False)        
# ### ---1
# state_dict = {}
# for k, v in checkpoint['state_dict_'].items():
#     name = k.replace('.module', '')
#     print(name) 
#     state_dict[name] = v
# model.load_state_dict(state_dict, True)


# ### ---2
# # for k, v in checkpoint['state_dict_'].items():
# #     name = k.replace('.module', '')
# #     print(name)
# #     if name == 'classifier.1.weight':
# #         name = 'classifier.0.weight'
# #     if name == 'classifier.1.bias':
# #         name = 'classifier.0.bias'
# #     state_dict[name] = v
# # model.load_state_dict(state_dict, True)



# torch_output = model(img)

# export_onnx_file = "model_large_2_v2_softmax_dim1_V2_dropout1.onnx"
 
# x=torch.onnx.export(model,                                  
#                 img,                                      
#                 export_onnx_file,                         
#                 verbose=False,                            
#                 input_names=["input"],                  
#                 output_names=["output"],                   
#                 opset_version=10,                        
#                 do_constant_folding=True,                
#                 # dynamic_axes={"input":{0: "batch_size"}, "output":{0: "batch_size"},} 
#                 )                                          

# net = onnx.load("model_large_2_v2_softmax_dim1_V2_dropout1.onnx")   
# onnx.checker.check_model(net)                                   
# session = onnxruntime.InferenceSession("./model_large_2_v2_softmax_dim1_V2_dropout1.onnx")  
# out_r = session.run(None, {"input": img.cpu().numpy()})                                    
   
# print('*'*50)
# print(torch_output)
# print(out_r)
################################################################# 

# # V2      yangxiaohui ----- help 

# 分析
# mv3_resnet50_large_test_1img.py
# mv3_resnet50_large_test_1img_o.py
# 发现加载网络时  from mv3_resnet50_large_2 import mobilenetv3
# 网络中 nn.Dropout(p=dropout) 不能被注释掉   但其实net.eval()会自己处理这些  包括BN层等等   而且确实要添加softmax()
# pth测试ok

# 修改1
# 所以在   torch2onnx.py中不能单纯只在 mv3_resnet50_large_2.py 处理注释dropout层  
# 而是要在 torch2onnx.py中 model  = mobilenetv3().cuda().eval()  本质上解决测试和训练中的区别

# 修改2
# 此外    torch2onnx.py中可以测试onnx是否转换成功  测onnx模型
# 测试用例 out_r = session.run(None, {"input": img.cpu().numpy()})  对onnx的session()输入图像--给实例化
# onnx 测试ok
 
# 修改3
# 代码方面  对于pth_onnx转化过程   下面代码三种

# # model.load_state_dict(checkpoint['state_dict_'], False)
# 第一种  也就是之前用法  结果是概率置信度一样   其实就是模型根本没有加载成功   

# # model.load_state_dict(checkpoint['state_dict_'], True)
# 改成True  保证严格对应   发现就加载不了了 
# 因为训练是多机保存的模型有额外的关键字module    测试时单机需要去掉   
# 也就是下面的读取方式   for k, v in checkpoint['state_dict_'].items()
# 报错是 
# features.module.0.0.weight    features.0.0.weight
# features.module.0.1.weight    features.0.1.weight
# ......                        ...... 
 
# # state_dict = {}
# # for k, v in checkpoint['state_dict_'].items():
# #     name = k.replace('.module', '')
# #     print(name) 
# #     state_dict[name] = v
# # model.load_state_dict(state_dict, True)
  
# 至于还有下面的删除 if name == 'classifier.1.weight':  name == 'classifier.0.weight' 
# 是为了针对删除网络的dropout  但其实没必要  直接eval()根治
  
# 多机的模型单机测试  加载模型问题  直接加载 True 不行  多了关键字   False 能加载但其实是虚的  
# 至于 mv3_resnet50_large_test_1img.py  True  False  都能加载对模型 
# 还是因为  81行  False的时候  其实用的是多机测试加载  
################################################################# 
  
# import sys
# import os 
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))
# import onnx 
# import torch
# from mv3_resnet50_large_2 import mobilenetv3, MobileNetV3
# import cv2
# import onnxruntime
# import numpy as np
 
# data_root ='/search/odin/likun/test_6/8pad_imgs/1005.jpg_02.jpg'
# img       = torch.from_numpy(cv2.resize(cv2.imread(data_root), (300, 300))[np.newaxis, :, :, :].astype(np.float32))
# img       = img.permute(0, 3, 1, 2).cuda() / 255.
 
# # model   = mobilenetv3.cuda
# model     = mobilenetv3().cuda().eval()    ####  .eval() .eval()  
 
# checkpoint = torch.load("/search/odin/likun/zyresnet/A_onnx/model_large_2_transfer.pth")
 
# ### ---0
# # model.load_state_dict(checkpoint['state_dict_'], False)  
# # model.load_state_dict(checkpoint['state_dict_'], True)       
 
# ### ---1
# state_dict = {}
# for k, v in checkpoint['state_dict_'].items():
#     name = k.replace('.module', '')
#     print(name) 
#     state_dict[name] = v
# model.load_state_dict(state_dict, True)

# ###
# ### ---2
# # for k, v in checkpoint['state_dict_'].items():
# #     name = k.replace('.module', '')
# #     print(name)
# #     if name == 'classifier.1.weight':
# #         name = 'classifier.0.weight'
# #     if name == 'classifier.1.bias':
# #         name = 'classifier.0.bias'
# #     state_dict[name] = v
# # model.load_state_dict(state_dict, True)
 
# torch_output = model(img)

# export_onnx_file = "model_large_2_v2_softmax_dim1_V2_dropout2.onnx"
 
# x=torch.onnx.export(model,                                 
#                 img,                                     
#                 export_onnx_file,                          
#                 verbose=False,                           
#                 input_names=["input"],                    
#                 output_names=["output"],                  
#                 opset_version=10,                         
#                 do_constant_folding=True,                   
#                 # dynamic_axes={"input":{0: "batch_size"}, "output":{0: "batch_size"},} 
#                 )                                            

# net = onnx.load("model_large_2_v2_softmax_dim1_V2_dropout2.onnx")    
# onnx.checker.check_model(net)                                         
 
# session = onnxruntime.InferenceSession("./model_large_2_v2_softmax_dim1_V2_dropout2.onnx") 
# out_r = session.run(None, {"input": img.cpu().numpy()})                                   
   
# print('*'*50)
# print(torch_output)
# print(out_r)
 
#################################################################  
# 根据模型训练时的测试集处理方法   进一步归一化 仿射  处理 比如
# img[:, 0, :, :] = (img[:, 0, :, :] - 0.485)/0.229 
 
import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))
import onnx 
import torch
from mv3_resnet50_large_2 import mobilenetv3, MobileNetV3
import cv2
import onnxruntime
import numpy as np
 
data_root ='/search/odin/likun/test_6/8pad_imgs/WechatIMG2640.jpeg'   #    1005.jpg_02.jpg   WechatIMG2640.jpeg
img       = torch.from_numpy(cv2.cvtColor(cv2.resize(cv2.imread(data_root), (300, 300)), cv2.COLOR_BGR2RGB)[np.newaxis, :, :, :].astype(np.float32))
img       = img.permute(0, 3, 1, 2) / 255.
img[:, 0, :, :] = (img[:, 0, :, :] - 0.485)/0.229
img[:, 1, :, :] = (img[:, 1, :, :] - 0.456)/0.224
img[:, 2, :, :] = (img[:, 2, :, :] - 0.406)/0.225
  
# model   = mobilenetv3.cuda 
model     = mobilenetv3().cuda().eval()   
 
checkpoint = torch.load("/search/odin/likun/zyresnet/A_onnx/model_large_2_transfer.pth")
 
### ---0
# model.load_state_dict(checkpoint['state_dict_'], False)  
# model.load_state_dict(checkpoint['state_dict_'], True)          
 
### ---1
state_dict = {}
for k, v in checkpoint['state_dict_'].items():
    name = k.replace('.module', '')
    print(name) 
    state_dict[name] = v
model.load_state_dict(state_dict, True)
 
### ---2
# for k, v in checkpoint['state_dict_'].items():
#     name = k.replace('.module', '')
#     print(name)
#     if name == 'classifier.1.weight':
#         name = 'classifier.0.weight'
#     if name == 'classifier.1.bias':
#         name = 'classifier.0.bias'
#     state_dict[name] = v
# model.load_state_dict(state_dict, True)
 
torch_output = model(img.cuda())

export_onnx_file = "model_large_2_v2_softmax_dim1_V2_dropout2_300.onnx"
   
x=torch.onnx.export(model,                               
                img.cuda(),                               
                export_onnx_file,                          
                verbose=False,                             
                input_names=["input"],                     
                output_names=["output"],                   
                opset_version=10,                         
                do_constant_folding=True,                  
                # dynamic_axes={"input":{0: "batch_size"}, "output":{0: "batch_size"},} 
                )                                         

net = onnx.load("model_large_2_v2_softmax_dim1_V2_dropout2_300.onnx")    
onnx.checker.check_model(net)                                         
 
session = onnxruntime.InferenceSession("./model_large_2_v2_softmax_dim1_V2_dropout2_300.onnx")  
out_r = session.run(None, {"input": img.cpu().numpy()})                                 
   
print('*'*50) 
print(out_r) 
print(torch_output)

 

