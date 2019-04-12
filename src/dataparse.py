import numpy as np
import pandas as pd
import random
# import matplotlib
# matplotlib.use('Agg')


def read_extract_data(input_path, debug=True):
    """Read mobile phone data 

    Args:
        input_path (str): directory to nasdaq dataset.

    Returns:
        X (np.ndarray): features.
        y (np.ndarray): ground truth.

    """
    df = pd.read_csv(input_path, nrows = 250 if debug else None)
    X = df.iloc[:, 0:-1].values
    y = df.iloc[:, -1].values
    

    """"generate mask """
   
    X_last=np.zeros_like(X)
    X_last[0,:]=X[0,:]
    count=0

    for i in range(1,X.shape[0]):
        for j in range(1,X.shape[1]):

            if(random.randint(1,1000)<0):
                X[i,j]=0
                #print(count)
        
    

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):

            if(X[i,j]==0):
                X_last[i,j]=X_last[i-1,j]
                #print("count")
            else:
                X_last[i,j]=X[i,j]

    """generate delta"""
    """generate X_last"""

    return X, y, X_last
