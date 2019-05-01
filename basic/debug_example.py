import torch                                                                    
import torch.nn as nn                                                           
                                                                                
class MyNet(torch.nn.Module):                                                   
  def __init__(self, n_in, n_h, n_out):                                         
    super(MyNet, self).__init__()                                               
    self.linear1 = torch.nn.Linear(n_in, n_h)                                   
    self.linear2 = torch.nn.Linear(n_h, n_out)                                  
    import pdb                                                                  
    pdb.set_trace()                                                             
                                                                                
  def forward(self, x):                                                         
    h_relu = self.linear1(x).clamp(min=0)                                       
    y_pred = self.linear2(h_relu)                                               
    return y_pred                                                               
                                                                                
n_in, n_h, n_out, batch_size = 100, 64, 10, 32                                  
                                                                                
model = MyNet(n_in, n_h, n_out)  
