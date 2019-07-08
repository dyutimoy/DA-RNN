"""
Main pipeline of DA-RNN.

@author Zhenye Na 05/21/2018

"""

import torch
import argparse
import numpy as np
import pandas as pd
from torch import nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from platypus.core import Problem
from platypus.types import Real
from platypus.algorithms import NSGAII, SMPSO, OMOPSO

# from tqdm import tqdm
from torch.autograd import Variable

from ops import *

from model import *
from dataparse import *

# Parameters settings
parser = argparse.ArgumentParser(description="DA-RNN")

# Dataset setting
parser.add_argument('--dataroot', type=str, default="../phone/phoneDatasetFinal.csv", help='path to dataset')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers [2]')
parser.add_argument('--batchsize', type=int, default=128, help='input batch size [128]')

# Encoder / Decoder parameters setting
parser.add_argument('--nhidden_encoder', type=int, default=128, help='size of hidden states for the encoder m [64, 128]')
parser.add_argument('--nhidden_decoder', type=int, default=128, help='size of hidden states for the decoder p [64, 128]')
parser.add_argument('--ntimestep', type=int, default=10, help='the number of time steps in the window T [10]')

# Training parameters setting
parser.add_argument('--epochs', type=int, default=5000, help='number of epochs to train [10, 200, 500]')
parser.add_argument('--resume', type=bool, default=False, help='resume training or not')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate [0.001] reduced by 0.1 after each 10000 iterations')
parser.add_argument('--ngpu', type=int, default=0, help='number of GPUs to use')
parser.add_argument('--cuda', action='store_true', help='s')
parser.add_argument('--debug',type=bool,default=False,help='debug code')
parser.add_argument('--name',type=int,default=1,help="version")
parser.add_argument('--manualSeed', type=int, help='manual seed')
opt = parser.parse_args()

print(opt)
# Read dataset
X, y,X_last = read_extract_data(opt.dataroot, debug=opt.debug)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize model
model = DA_rnn(X, y,X_last, opt.ntimestep, opt.nhidden_encoder, opt.nhidden_decoder, opt.batchsize, opt.lr, opt.epochs,opt.resume)
model=model.to(device)
# Train
#model.train()




#x_input=np.array([-1,5.1,144,1.5,5,13,16])
# Prediction
#y_pred = model.test()
def mobile_phone(vars):
    x1=vars[0]
    x2=vars[1]
    x3=vars[2]
    x4=vars[3]
    x5=vars[4]
    x6=vars[5]    
    x7=vars[6]
    x_input=np.array([x1,x2,x3,x4,x5,x6,x7])
    min1=0
    min2=1.5
    min3=50
    min4=0.4
    min5=1.2
    min6=0.9
    min7=0
    
    max1=4
    max2=7.1
    max3=2005
    max4=2.7
    max5=9.7
    max6=24
    max7=64
    val=model.evalModel(x_input)
    print(val)
    return [val,((max1-x1)/(max1-min1))+((max2-x2)/(max2-min2))+((max3-x3)/(max3-min3))+((max4-x4)/(max4-min4))+((max5-x5)/(max5-min5))+((max6-x6)/(max6-min6))+((max7-x7)/(max7-min7))]
    

problem1=Problem(7,2,0)
problem1.types[:]=[Real(0,4),Real(1.5,7.1),Real(50,2005),Real(0.4,2.7),Real(1.2,9.7),Real(0.9,24),Real(0,64)]

problem1.function=mobile_phone

#algorithm1=NSGAII(problem1)
#algorithm2=SMPSO(problem1)

algorithm3=OMOPSO(problem1,epsilons=0.05)

#algorithm1.run(2)
#algorithm2.run(2)

algorithm3.run(50)

#for solution in algorithm1.result:
#    print(solution.objectives)
    
for solution in algorithm3.result:
    print(solution.objectives)
    
#for solution in algorithm3.result:
#  print(solution.objectives)


"""
y_pred = model.test()
print(y_pred)
fig1 = plt.figure()
plt.semilogy(range(len(model.iter_losses)), model.iter_losses)
numpy.savetxt("iter_loss"+str(opt.name)+".csv", model.iter_losses, delimiter=",")
plt.savefig("1_"+str(opt.name)+".png")
plt.close(fig1)

fig2 = plt.figure()
plt.semilogy(range(len(model.epoch_losses)), model.epoch_losses)
numpy.savetxt("epoch_loss"+str(opt.name)+".csv",  model.epoch_losses, delimiter=",")
plt.savefig("2_"+str(opt.name)+".png")
plt.close(fig2)def mobile_phone(vars):
    x1=vars[0]
    x2=vars[1]
    x3=vars[2]
    x4=vars[3]
    x5=vars[4]
    x6=vars[5]    
    x7=vars[6]
    x_input=np.array([x1,x2,x3,x4,x5,x6,x7])
    min1=0
    min2=1.5
    min3=50
    min4=0.4
    min5=1.2
    min6=0.9
    min7=0
    
    max1=4
    max2=7.1
    max3=2005
    max4=2.7
    max5=9.7
    max6=24
    max7=64
    
    return [model.evalModel(x_input),[((max1-x1)/(max1-min1))+((max2-x2)/(max2-min2))+((max3-x3)/(max3-min3))+((max4-x4)/(max4-min4))+((max5-x5)/(max5-min5))+((max6-x6)/(max6-min6))+((max7-x7)/(max7-min7))]

fig3 = plt.figure()
plt.plot(y_pred, label = 'Predicted')
plt.plot(model.y[model.train_timesteps:], label = "True")
plt.legend(loc = 'upper left')
plt.savefig("3_"+str(opt.name)+".png")
plt.close(fig3)
print('Finished Training')

"""