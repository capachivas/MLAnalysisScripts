
from datetime import timezone
from datetime import datetime
from datetime import date
import array as arr
import pandas as pd
import numpy as np
import numpy.matlib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import matplotlib.cm as cm
import matplotlib.colors as colors
from sklearn.manifold import TSNE
import seaborn as sns
import math
import scipy


def binarize_signal(y):

    #normalized signal
    
    y=y/max(y)
    
    threshold=max(y)/6 #chosen to make d=0 in second test
    
    #raw input 
    fig,failed =  plt.subplots(1,figsize=(16,10))

    failed.plot(y)
    
    #normalized signal
    y_output=np.zeros(len(y))

    #thresholded signal (based on MSE threshold error to discern)

    y_output[(y>threshold)]=1

    y_output[(y<=threshold)]=0
    
     #raw input 
    fig,failed =  plt.subplots(1,figsize=(16,10))

    failed.plot(y_output)
    
    return y_output

#pending represent features names
df_training_baseline=pd.read_csv('0.8Test_mean_squared_logarithmic_error_1Layer80.csv')  
y0= binarize_signal((df_training_baseline.iloc[:,1]).values)

print(scipy.spatial.distance.jaccard(y0,y0))
print(scipy.spatial.distance.cosine(y0,y0))

df_test1=pd.read_csv('IncTest_mean_squared_logarithmic_error_1Layer80.csv')  
y1= binarize_signal((df_test1.iloc[:,1]).values)

print(scipy.spatial.distance.jaccard(y0,y1))
print(scipy.spatial.distance.cosine(y0,y1))


df_test2=pd.read_csv('0.2Test_mean_squared_logarithmic_error_1Layer80.csv')  
y2= binarize_signal((df_test2.iloc[:,1]).values)

print(scipy.spatial.distance.jaccard(y0,y2))
print(scipy.spatial.distance.cosine(y0,y2))

 