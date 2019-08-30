import numpy as np
import pandas as pd
import random
# import matplotlib
# matplotlib.use('Agg')


def read_extract_data(input_path, debug=False):
    """Read mobile phone data 

    Args:
        input_path (str): directory to nasdaq dataset.

    Returns:
        X (np.ndarray): features.
        y (np.ndarray): ground truth.

    """
    df = pd.read_csv(input_path, nrows = 100 if debug else None)
    X = df.iloc[:, 0:-1].values
    y = df.iloc[:, -1].values
    
    
    """"generate mask """
   
    X_last=np.zeros_like(X)
    y_norm=np.zeros((X.shape[0],),dtype=float)
    
    X_last[0,:]=X[0,:]
    count=0
    """
    for i in range(1,X.shape[0]):
        for j in range(1,X.shape[1]):

            if(random.randint(1,1000)<0):
                X[i,j]=0
                #print(count)
        
    """


    for j in range(X.shape[1]):
        max_val=0.0
        min_val=99999999.9999
        for i in range(X.shape[0]):
            if(X[i,j]!=-1 and X[i,j]<min_val):
                min_val=X[i,j]

            if(X[i,j]!=-1 and X[i,j]>max_val):
                max_val=X[i,j]
        den=max_val-min_val            
        for ix in range(X.shape[0]):
            if(X[ix,j]!=-1):
                X[ix,j]=(X[ix,j]-min_val)/den


         
    max_valy=0.0
    min_valy=99999999.99      
    for i in range(X.shape[0]):
        
        if( y[i]<min_valy):
            min_valy=y[i]

        if( y[i]>max_valy):
            max_valy=y[i]
        
    den_y=max_valy-min_valy    
    #Sprint(max_valy-min_valy)
    for i in range(X.shape[0]):
        
        a=(y[i]-min_valy)/den_y
        y_norm[i]=float(a)
        #print(y_norm[i])
        
                              


    #np.savetxt("hi.csv", X, delimiter=",")
    #np.savetxt("hssi.csv", y_norm, delimiter=",")

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):

            if(X[i,j]==-1):
                X_last[i,j]=X_last[i-1,j]
                #print("count")
            else:
                X_last[i,j]=X[i,j]
    

    """generate delta"""
    """generate X_last"""

    return X, y_norm, X_last
