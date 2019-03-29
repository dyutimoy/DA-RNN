"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch import optim
from torch.autograd import Variable


#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#print(device)
#print("hi")


x = torch.randn(5,3,2)            # Size 2x3
#y = x.view(9)                    # Resize x to size 6
#z = x.view(-1, 3) 
#=Variable(x.data.new(2,2,2).zero_())

print(x)
#print(y)
#print(z)
#print(w)
h_n=Variable(x.data.new(
            1, x.size(0),2).zero_())
s_n=Variable(x.data.new(
            1, x.size(0), 2).zero_())            
s_d =h_n.repeat(2, 2, 2).permute(1, 0, 2)
print(s_d)
a = torch.cat((h_n.repeat(2, 1, 1).permute(1, 0, 2),
                           s_n.repeat(2, 1, 1).permute(1, 0, 2),
                           x.permute(0, 2, 1)), dim=2)
print(a)   
b=a.view(-1,7)
print(b) 
a=a.view(-1,7).view(-1,2)
print(a) 
"""
import torch
import pycuda.driver as cuda
cuda.init()
print(torch.cuda.current_device())
print(cuda.Device(0).name())
print(torch.cuda.memory_allocated())
print(torch.cuda.memory_cached())