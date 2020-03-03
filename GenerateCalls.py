
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

#f = open("startTimes.txt", "r")


#with open('startTimes.txt') as f:
#    starts = f.read().strip('\n')

starts=np.loadtxt("startTimes.txt")  
n_samples=len(starts) 

#mu is the average in seconds
mins_required=3
mu, sigma = mins_required*60, 0.10 # mean and standard deviation 

def GenerateDurations(n_samples,sigma,mu):
    #fig,failed =  plt.subplots(1,figsize=(16,10))
    s = np.random.normal(mu, sigma, n_samples)
    #count, bins, ignored = plt.hist(s, 30, normed=True)
    #failed.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='r')
    return s

durations=(GenerateDurations(n_samples,sigma,mu))
ends=starts+durations

#ends=str(ends).lstrip('[').rstrip(']')
#starts=str(starts).lstrip('[').rstrip(']')

ends=(np.vstack(ends))
starts=(np.vstack(starts))

fig,failed =  plt.subplots(1,figsize=(16,10))
count, bins, ignored = plt.hist(durations, 30, normed=True)
failed.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='r')


np.savetxt('startTimes2.txt', np.c_[starts,ends],delimiter=';',fmt='%i')
