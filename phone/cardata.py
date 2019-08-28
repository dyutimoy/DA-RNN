

import sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
#import matplotlib.pyplot as plt

import csv
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, make_scorer, mean_squared_error

# If true, include make and model in Random Forest
# Shows how much make and model come into play, but when when we calculate prices
# we should omit to see if the models are overpriced
includeMakeAndModel = True



def GetDataMatrix():
    
    # Data frame with make and model
    Xmodelmake = pd.read_csv("./data.csv",header=0, usecols=(0,2,3,4,5,6,7,8,9,10,11,12,13,));
    Z=pd.read_csv("./data.csv",header=0, usecols=(9,));
    X = Xmodelmake
    Y = pd.read_csv("./data.csv",header=0, usecols=(14,));

    #X, Y, Xmodelmake = shuffle(X, Y, Xmodelmake)
    
    
    # Turns categorical data into binary values across many columns    
    X = pd.get_dummies(X, dummy_na = False, columns=['Make', 'Engine Fuel Type', 'Transmission Type', 'Driven_Wheels', 'Market Category', 'Vehicle Size', 'Vehicle Style'] );
   
    # Fill the null values with zeros
    X.fillna(-1, inplace=True);
    return (X, Y, Z,Xmodelmake)




(X, Y,Z, Xmodelmake) = GetDataMatrix() #Gets the X,Y

# Turn into a proper one D arrayY = numpy.ravel(Y);
Y_unraveled = np.ravel(Y);


pd.DataFrame(X).to_csv("car_fea.csv")


pd.DataFrame(Y).to_csv("car_price.csv")

A=np.zeros((Z.shape[0],10))
m1="Factory Tuner"
m2="Luxury"
m3="High-Performance"
m4="Performance"
m5="Flex Fuel"
m6="Hatchback"
m7="Diesel"
m8="Crossover"
m9="Exotic"
m10="Hybrid"
i=0



Z = Z.values
Z_new=Z.astype(str)
print(Z_new[0])


print(Z_new.dtype)

for i in range(Z_new.shape[0]) :
    #temp=Z_new[i]
    #print("hi:",temp)
    #result = [x.strip() for x in Z_new[i].split(',')]
    result = np.char.split(Z_new[i], sep =',')
    #print("sddddd",out_arr) 
    #print(result[0])
    for j in range(len(result[0])):
        
        #print( m6==result[0][j])
        
        
        if result[0][j] == m1:
            A[i,0]=1
            print( m1==result[0][j])
        elif result[0][j] == m2:
            A[i,1]=1
            print( m2==result[0][j])
        elif result[0][j] == m3:
            A[i,2]=1
            print( m3==result[0][j])
        elif result[0][j] == m4:
            A[i,3]=1
            print( m4==result[0][j])
        elif result[0][j] == m5:
            A[i,4]=1
            print( m5==result[0][j])
        elif result[0][j] == m6:
            A[i,5]=1
            print( m6==result[0][j])
        elif result[0][j] == m7:
            A[i,6]=1
            print( m7==result[0][j])
        elif result[0][j] == m8:
            A[i,7]=1
            print( m8==result[0][j])
        elif result[0][j] == m9:
            A[i,8]=1
            print( m9==result[0][j])
        elif result[0][j] == m10:
            A[i,9]=1
            print( m10==result[0][j])
        elif result[0][j] == "nan":
            print( "naaaaaaaaaaaaaaaaaan")    
        else:
            print("fucccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc")
            print(result[0][j])
            print("oooffffkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk")
         
    


#np.savetxt("carZ.csv", Z, delimiter=",")
np.savetxt("carA.csv", A, delimiter=",")