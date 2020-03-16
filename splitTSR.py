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

from statsmodels.tsa.seasonal import seasonal_decompose

import statsmodels.api as sm





#fig,failed =  plt.subplots(1,figsize=(16,10))



#failed.plot(y.dot(peak_position))



df= pd.read_csv("niort.csv", sep = ";", low_memory=False, error_bad_lines=False)





y=(df.loc[:,'container_memory_rss{container="",container_name="",id="/kubepods/besteffort/podd2703fdd-859e-4812-bdcd-282be0659c3e",image="",name="",namespace="openims",pod="hss-deployment-7b9c7786fd-qgvts",pod_name="hss-deployment-7b9c7786fd-qgvts"}'])



s=sm.tsa.seasonal_decompose(y,model='additive',freq=100)



fig,failed =  plt.subplots(1,figsize=(5,3))



failed.plot(s.trend[0:500])



fig,failed =  plt.subplots(1,figsize=(5,3))



failed.plot(s.seasonal[0:500])





fig,failed =  plt.subplots(1,figsize=(5,3))



failed.plot(s.resid[0:500])
