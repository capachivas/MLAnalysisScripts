
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

def column_index(dfx, query_cols):
    cols = dfx.columns.values
    sidx = np.argsort(cols)
    return sidx[np.searchsorted(cols,query_cols,sorter=sidx)]

df= pd.read_csv("niort_b4.csv", sep = ";", low_memory=False, error_bad_lines=False)

df = df.select_dtypes(exclude=['object'])
df.drop(list(df.filter(regex='Unnamed')),axis=1, inplace=True)
df.drop(list(df.filter(regex='timestamp')),axis=1, inplace=True)

regExString = 'icscf' # .* -> all features, change for use specific group of features

df_pcscf = df.filter(regex=regExString)

stdBoolA = df_pcscf.std() < 0.00000001 # we only consider features with std > 0
features_static = df_pcscf.columns[stdBoolA == True]

indexes_features_static_b4=((column_index(df_pcscf, features_static)))


df= pd.read_csv("niort_b.csv", sep = ";", low_memory=False, error_bad_lines=False)

df = df.select_dtypes(exclude=['object'])
df.drop(list(df.filter(regex='Unnamed')),axis=1, inplace=True)
df.drop(list(df.filter(regex='timestamp')),axis=1, inplace=True)

regExString = 'icscf' # .* -> all features, change for use specific group of features

df_pcscf = df.filter(regex=regExString)

stdBoolA = df_pcscf.std() < 0.00000001 # we only consider features with std > 0
features_static = df_pcscf.columns[stdBoolA == True]

indexes_features_static_b1=column_index(df_pcscf, features_static)


df= pd.read_csv("niort_b3.csv", sep = ";", low_memory=False, error_bad_lines=False)

df = df.select_dtypes(exclude=['object'])
df.drop(list(df.filter(regex='Unnamed')),axis=1, inplace=True)
df.drop(list(df.filter(regex='timestamp')),axis=1, inplace=True)

regExString = 'icscf' # .* -> all features, change for use specific group of features

df_pcscf = df.filter(regex=regExString)

stdBoolA = df_pcscf.std() < 0.00000001 # we only consider features with std > 0
features_static = df_pcscf.columns[stdBoolA == True]

indexes_features_static_b3=column_index(df_pcscf, features_static)


df= pd.read_csv("niort_b2.csv", sep = ";", low_memory=False, error_bad_lines=False)

df = df.select_dtypes(exclude=['object'])
df.drop(list(df.filter(regex='Unnamed')),axis=1, inplace=True)
df.drop(list(df.filter(regex='timestamp')),axis=1, inplace=True)

regExString = 'icscf' # .* -> all features, change for use specific group of features

df_pcscf = df.filter(regex=regExString)

stdBoolA = df_pcscf.std() < 0.00000001 # we only consider features with std > 0
features_static = df_pcscf.columns[stdBoolA == True]

indexes_features_static_b2=column_index(df_pcscf, features_static)

print(len(indexes_features_static_b1))
print(len(np.intersect1d(indexes_features_static_b1,indexes_features_static_b2)))
print(len(np.intersect1d(np.intersect1d(indexes_features_static_b1,indexes_features_static_b2),indexes_features_static_b3)))
print(len(np.intersect1d(np.intersect1d(np.intersect1d(indexes_features_static_b1,indexes_features_static_b2),indexes_features_static_b3),indexes_features_static_b4)))
