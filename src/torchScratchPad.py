
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch import optim
from torch.autograd import Variable


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(device)
#print("hi")


x = torch.randn(5,3,2)            # Size 2x3
#y = x.view(9)                    # Resize x to size 6
#z = x.view(-1, 3) 
#=Variable(x.data.new(2,2,2).zero_())

print(x)
#print(y)
#print(z)
#print(w)
mask = Variable(x.data.new(
            x.size(0), 3, 2).zero_()).cuda()
       
deltaX= Variable(x.data.new(
            1,x.size(0),2).zero_()).cuda()
print(deltaX)
print(deltaX.size())
print("breaks")
deltaX=torch.squeeze(deltaX)            
print(deltaX)
print(deltaX.size())
mask[1,2,1]=8
print(mask)