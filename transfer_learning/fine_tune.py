# pytorch fine tune example                                                     
import torch                                                                    
import torchvision.models as models                                             
from torchvision import transforms                                              
import torch.nn as nn                                                           
import torch.optim as optim                                                     
from PIL import Image                                                           
import torch.nn.functional as F                                                 
                                                                                
model = models.resnet18(pretrained=True)                                        
                                                                                
# freeze the parameters so that the gradients are not computed in backward().   
for param in model.parameters():                                                
    param.requires_grad = False                                                 
                                                                                
# model last layer                                                              
num_ftrs = model.fc.in_features                                                 
model.fc = nn.Linear(num_ftrs, 2)                                               
                                                                                
# we can add more layers here by using add_module                               
# for example: ("my_conv", Conv2d(3, 16, 5, padding=2))                         
model.add_module("my_fc_2",nn.Linear(2,1))                                      
                                                                                
# only parameters of two layer are being optimized                              
# we concatenate those two parameter lists                                      
params = list(model.fc.parameters()) + list(model.my_fc_2.parameters())         
optimizer = optim.SGD( params, lr=0.001, momentum=0.9)                          
                                                                                
print( params )                                                                 
                                                                               
# normal pytorch code like get data, training process etc ....                  
# skip here                                                                     
                                                                                
#############################################################################   
# second pass                                                                   
# unfreeze the parameters                                                       
for param in model.parameters():                                                
    param.requires_grad = True                                                  
                                                                                
# for debug/demo you can turn it on                                             
# params is a generator                                                         
params = model.parameters()                                                     
# turn it to be a list                                                          
lst = list(params)                                                              
print(lst)                                                                      
# all parameters are being optimized                                            
optimizer_all = optim.SGD(lst, lr=0.001, momentum=0.9)                          
                                                                                
# or just pass model.parameters() to it                                         
#optimizer_all = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)          
                                                                                
# normal pytorch code like get data, training process etc ....                  
# skip here                                  
