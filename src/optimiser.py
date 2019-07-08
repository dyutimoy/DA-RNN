from platypus.core import Problem
from platypus.types import Real
from platypus.algorithms import NSGAII, SMPSO, OMOPSO
from model import *

def mobile_phone(vars,model):
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