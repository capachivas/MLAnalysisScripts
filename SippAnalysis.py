# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 13:33:33 2020

@author: Alessio Diamanti
"""
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



def mon(df):
    new_df=df.copy()
    x=new_df.values
    candidates=np.diff(x,axis=0)>=0
    indexes_bool=np.all(candidates,axis=0)
    features_inc_monotonic=df.columns[indexes_bool==True]
    new_df[features_inc_monotonic] = new_df[features_inc_monotonic].diff()
    return new_df


def select_group_features(df,string):
    #make it case sensitive
    #gest a group of features based on their name on string and gets also summation and product on those group 
    m = np.core.defchararray.find(df.columns.values.astype(str),string) >= 0
    new_df=(pd.DataFrame(df.values[:, m], df.index, df.columns[m]))
    
    if(new_df.shape[1]>0): #the new dataframe got some information based on the string given as input

        new_df['average']=np.average(new_df,axis=1)
        new_df['multiplication']=np.prod(new_df[new_df.columns[~new_df.columns.isin(['average'])]],axis=1)
        new_df['summation']=np.sum(new_df[new_df.columns[~new_df.columns.isin(['average','multiplication'])]],axis=1)
       
        return new_df
    else: #if the string does not correspond to anything we will not make any operation
        return -1
        print("not result found  based on string: "+str(string))
        
def strptime(val):
    if '.' not in val:
        return datetime.datetime.strptime(val, "%Y-%m-%dT%H:%M:%S")

    nofrag, frag = val.split(".")
    date = datetime.datetime.strptime(nofrag, "%Y-%m-%dT%H:%M:%S")

    frag = frag[:6]  # truncate to microseconds
    frag += (6 - len(frag)) * '0'  # add 0s
    return date.replace(microsecond=int(frag))     


infra_rate=5
sipp_rate=60

resampling_rate=int(sipp_rate/infra_rate)

df_sipp= pd.read_csv("sipp_statistics30.csv", sep = ";", low_memory=False, error_bad_lines=False)

df_sipp=df_sipp[['CurrentTime','FailedCall(P)']]

df_infra= pd.read_csv("physicalNiort30.csv", sep = ";", low_memory=False, error_bad_lines=False)

df_infra.drop(list(df_infra.filter(regex='Unnamed')),axis=1, inplace=True) #regex

#replacing current timestamp by integer timestamp

df_sipp.CurrentTime = df_sipp.CurrentTime.str.replace(r'Z', '')
df_sipp.CurrentTime = df_sipp.CurrentTime.str.replace(r'T', ' ')
df_sipp.CurrentTime = df_sipp.CurrentTime.str.split('.').str[0]


hour_of_day = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").hour
minute_of_day = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").minute
second_of_day = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").second

df_sipp.CurrentTime=60*df_sipp.CurrentTime.map(minute_of_day) + 60*60*(df_sipp.CurrentTime.map(hour_of_day))+(df_sipp.CurrentTime.map(second_of_day))

df_infra.timestamp = df_infra.timestamp.str.replace(r'Z', '')
df_infra.timestamp = df_infra.timestamp.str.replace(r'T', ' ')
df_infra.timestamp = df_infra.timestamp.str.split('.').str[0]

region_hour_of_day = lambda x: hours_of_day[datetime.strptime(x, "%H:%M:%S").hour]
hour_of_day = lambda x: datetime.strptime(x, "%H:%M:%S").hour
minute_of_day = lambda x: datetime.strptime(x, "%H:%M:%S").minute
second_of_day = lambda x: datetime.strptime(x, "%H:%M:%S").second

df_infra.timestamp=60*df_infra.timestamp.map(minute_of_day) + 60*60*(df_infra.timestamp.map(hour_of_day))+(df_infra.timestamp.map(second_of_day))

df_infra = df_infra.select_dtypes(exclude=['object'])

start_time_sipp=df_sipp['CurrentTime'][0]

end_time_sipp=df_sipp['CurrentTime'][df_sipp.shape[0]-1]

start_time_infra=df_infra['timestamp'][0]

end_time_infra=df_infra['timestamp'][df_infra.shape[0]-1]

Lsipp=df_sipp.shape[0]

Linfra=df_infra.shape[0]

#getting the common integer timestamp


start_time=max(start_time_sipp,start_time_infra)

end_time_infra_corrected=start_time_infra+Linfra*infra_rate

end_time_sipp_corrected=start_time_sipp+Lsipp*sipp_rate

end_time=min(end_time_sipp_corrected,end_time_infra_corrected)

print(start_time)

print(end_time)


df_infra=df_infra[(df_infra.timestamp>=start_time) & (df_infra.timestamp<=end_time)]

df_sipp=df_sipp[(df_sipp.CurrentTime>=start_time) & (df_sipp.CurrentTime<=end_time)]


#rewriting the timestamps according to sipp_rate and infra_rate
df_sipp.CurrentTime=[start_time+sipp_rate*(k) for k in range(0,df_sipp.shape[0])]

df_infra.timestamp=[start_time+infra_rate*(k) for k in range(0,df_infra.shape[0])]


number_of_samples_sipp=((1+math.floor((end_time-start_time)/sipp_rate)))

number_of_samples_infra=((1+math.floor((end_time-start_time)/infra_rate)))

print(number_of_samples_sipp)

print(number_of_samples_infra)


print(df_infra.shape)

df_infra=df_infra.loc[0:number_of_samples_infra-1,:]

print(df_infra.shape)

df_sipp=df_sipp.loc[0:number_of_samples_sipp-1,:]

print(df_sipp.shape)

print(df_infra.shape)


#resampling
x=df_sipp['FailedCall(P)'].values


plt.subplots(1,figsize=(16,10))

plt.stem(np.cumsum(x))

print(x.shape)
y=np.zeros(resampling_rate)
y[0]=1

y=np.matlib.repmat(y,1,number_of_samples_sipp)

print(df_infra.shape[0])

y=y[0]


print(len(y))

y=y.reshape(-1,x.shape[0])
print(len(x))
z=np.multiply(y,x)
print(z.shape)
plt.subplots(1,figsize=(16,10))

print(z.shape)
z=z[0:df_infra.shape[0]]
print(z.shape)

plt.stem(np.cumsum(z))

pd.concat([df_infra, pd.DataFrame(columns = [ 'service_failed'])])


df_sipp2=pd.DataFrame(data=z,columns=['FailedCall'])

print(df_infra.shape)
print(df_sipp.shape)

df_final=pd.concat([df_infra,df_sipp2])
print(df_final)
df_final.to_csv("go.csv",sep=";")

fig,failed2 =  plt.subplots(1,figsize=(16,10))
failed2.plot(df_sipp.CurrentTime,np.cumsum(df_sipp['FailedCall(P)']))
failed2.set_title(str( 'Cumulative Failed Rate without resampling'), fontdict ={'verticalalignment': 'baseline'})

fig,failed2 =  plt.subplots(1,figsize=(16,10))
failed2.plot(df_infra.timestamp,np.cumsum(z))
failed2.set_title(str( 'Cumulative Failed Rate resampled'), fontdict ={'verticalalignment': 'baseline'})

#df_sipp2=pd.DataFrame(data=, columns=['timestamp','failed'])

#df_infra.drop(list(df_infra.filter(regex='Unnamed')),axis=1, inplace=True) #regex
#df_infra = df_infra.select_dtypes(exclude=['object']) ## for the physical part containing machine name

#df_infra.to_csv("hi.csv",sep = ";")

#representation of initial no resampling dataset
#fig,failed =  plt.subplots(1,figsize=(16,10))
#new_df=select_group_features(df,'FailedCall(P)')
#xx=(new_df[new_df.columns[~new_df.columns.isin(['average','multiplication','summation'])]])
#failed.plot(np.cumsum(xx.values))
#failed.set_title(str( 'Cumulative Failed Rate without resampling'), fontdict ={'verticalalignment': 'baseline'})


#new_df=select_group_features(df_resampled,'FailedCall(P)')
#x=(new_df[new_df.columns[~new_df.columns.isin(['average','multiplication','summation'])]])
#x=(x.values)
#y=np.zeros(resampling_rate)
#y[0]=1

#y=np.matlib.repmat(y,1,n)
#y=y.reshape(x.shape[0],1)
#z=np.multiply(y,x)
#fig,failed2 =  plt.subplots(1,figsize=(16,10))
#fig,failed2 =  plt.subplots(1,figsize=(16,10))
#failed2.plot(np.cumsum(z))
#failed2.set_title(str( 'Cumulative Failed Rate resampled'), fontdict ={'verticalalignment': 'baseline'})


#TotalCallCreated=select_group_features(df,'TotalCallCreated')
#TT=TotalCallCreated[TotalCallCreated.columns[~TotalCallCreated.columns.isin(['average','multiplication','summation'])]]

#SuccessfullCall=select_group_features(df,'SuccessfulCall(P)')
#SS=SuccessfullCall[SuccessfullCall.columns[~SuccessfullCall.columns.isin(['average','multiplication','summation'])]]

#fig,failed =  plt.subplots(1,figsize=(16,10))
#ss=SS.values
#tt=TT.values
#failed.plot(100*ss/tt)
#failed.set_title(str('SucessFul VS Total'), fontdict ={'verticalalignment': 'baseline'})

#for l in range(0,31):
#
#    fig,failed =  plt.subplots(1,figsize=(16,10))
#    df= pd.read_csv("physicalRochelle10.csv", sep = ";", low_memory=False, error_bad_lines=False)
#    df.drop(list(df.filter(regex='Unnamed')),axis=1, inplace=True) #regex
#    df = df.select_dtypes(exclude=['object']) ## for the physical part containing machine name
#    df.dropna(axis='columns',inplace=True)
#    dfnew=mon(df)
#    #dfnew=df
#    new_df=select_group_features(dfnew,'node_cpu_seconds_total{cpu="'+str(l)+'",mode="system"}')
#    new_df=(new_df[new_df.columns[~new_df.columns.isin(['average','multiplication','summation'])]])
#    #cpu_scaling_frequency
#    print(new_df.describe())
#    failed.plot(new_df)
#    failed.plot(new_df.mean(axis=1))
#    failed.set_title(str('CPU 10'), fontdict ={'verticalalignment': 'baseline'})
#    
#    
#    import numpy as np
#    import matplotlib.pyplot as plt
#    import scipy.fftpack
#    
#    
#    fig,failed =  plt.subplots(1,figsize=(16,10))
#    
#    # Number of samplepoints
#    N = 600
#    # sample spacing
#    T = 1.0 / 3000000000
#    x = np.linspace(0.0, N*T, N)
#    
#    
#    for k in range(1,2):
#        y = new_df.values
#        yf = scipy.fftpack.fft(y)
#        xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
#        failed.plot(xf, 2.0/N * np.abs(yf[:N//2]))
#    
#    fig,failed =  plt.subplots(1,figsize=(16,10))
#    df= pd.read_csv("physicalRochelle20.csv", sep = ";", low_memory=False, error_bad_lines=False)
#    df.drop(list(df.filter(regex='Unnamed')),axis=1, inplace=True) #regex
#    df = df.select_dtypes(exclude=['object']) ## for the physical part containing machine name
#    df.dropna(axis='columns',inplace=True)
#    dfnew=mon(df)
#    new_df=select_group_features(dfnew,'node_cpu_seconds_total{cpu="'+str(l)+'",mode="system"}')
#    new_df=(new_df[new_df.columns[~new_df.columns.isin(['average','multiplication','summation'])]])
#    #cpu_scaling_frequency
#    print(new_df.describe())
#    failed.plot(new_df)
#    failed.plot(new_df.mean(axis=1))
#    failed.set_title(str('CPU 20'), fontdict ={'verticalalignment': 'baseline'})
#    
#    
#    
#    fig,failed =  plt.subplots(1,figsize=(16,10))
#    df= pd.read_csv("physicalRochelle40.csv", sep = ";", low_memory=False, error_bad_lines=False)
#    df.drop(list(df.filter(regex='Unnamed')),axis=1, inplace=True) #regex
#    df = df.select_dtypes(exclude=['object']) ## for the physical part containing machine name
#    df.dropna(axis='columns',inplace=True)
#    dfnew=mon(df)
#    new_df=select_group_features(dfnew,'node_cpu_seconds_total{cpu="'+str(l)+'",mode="system"}')
#    new_df=(new_df[new_df.columns[~new_df.columns.isin(['average','multiplication','summation'])]])
#    #cpu_scaling_frequency
#    print(new_df.describe())
#    failed.plot(new_df)
#    failed.plot(new_df.mean(axis=1))
#    failed.set_title(str('CPU 40'), fontdict ={'verticalalignment': 'baseline'})
#    
#
#
#
#
###niort
#for l in range(0,31):
#
#    fig,failed =  plt.subplots(1,figsize=(16,10))
#    df= pd.read_csv("physicalNiort10.csv", sep = ";", low_memory=False, error_bad_lines=False)
#    df.drop(list(df.filter(regex='Unnamed')),axis=1, inplace=True) #regex
#    df = df.select_dtypes(exclude=['object']) ## for the physical part containing machine name
#    df.dropna(axis='columns',inplace=True)
#    dfnew=mon(df)
#    #dfnew=df
#    new_df=select_group_features(dfnew,'node_cpu_seconds_total{cpu="'+str(l)+'",mode="user"}')
#    new_df=(new_df[new_df.columns[~new_df.columns.isin(['average','multiplication','summation'])]])
#    #cpu_scaling_frequency
#    print(new_df.describe())
#    failed.plot(new_df)
#    failed.plot(new_df.mean(axis=1))
#    failed.set_title(str('CPU 10'), fontdict ={'verticalalignment': 'baseline'})
#    
#
#    
#    
#    fig,failed =  plt.subplots(1,figsize=(16,10))
#    df= pd.read_csv("physicalNiort20.csv", sep = ";", low_memory=False, error_bad_lines=False)
#    df.drop(list(df.filter(regex='Unnamed')),axis=1, inplace=True) #regex
#    df = df.select_dtypes(exclude=['object']) ## for the physical part containing machine name
#    df.dropna(axis='columns',inplace=True)
#    dfnew=mon(df)
#    new_df=select_group_features(dfnew,'node_cpu_seconds_total{cpu="'+str(l)+'",mode="user"}')
#    new_df=(new_df[new_df.columns[~new_df.columns.isin(['average','multiplication','summation'])]])
#    #cpu_scaling_frequency
#    print(new_df.describe())
#    failed.plot(new_df)
#    failed.plot(new_df.mean(axis=1))
#    failed.set_title(str('CPU 20'), fontdict ={'verticalalignment': 'baseline'})
#    
#
#    
#    
#    fig,failed =  plt.subplots(1,figsize=(16,10))
#    df= pd.read_csv("physicalNiort40.csv", sep = ";", low_memory=False, error_bad_lines=False)
#    df.drop(list(df.filter(regex='Unnamed')),axis=1, inplace=True) #regex
#    df = df.select_dtypes(exclude=['object']) ## for the physical part containing machine name
#    df.dropna(axis='columns',inplace=True)
#    dfnew=mon(df)
#    new_df=select_group_features(dfnew,'node_cpu_seconds_total{cpu="'+str(l)+'",mode="user"}')
#    new_df=(new_df[new_df.columns[~new_df.columns.isin(['average','multiplication','summation'])]])
#    #cpu_scaling_frequency
#    print(new_df.describe())
#    failed.plot(new_df)
#    failed.plot(new_df.mean(axis=1))
#    failed.set_title(str('CPU 40'), fontdict ={'verticalalignment': 'baseline'})
#    
#   
#
#
# 
#
#    
#    
#####containers
#
#
#fig,failed =  plt.subplots(1,figsize=(16,10))
#df10= pd.read_csv("rochelle10.csv", sep = ";", low_memory=False, error_bad_lines=False)
#df10=mon(df10)
#
#print(df10)
#new_df10=select_group_features(df10,'sipp')
#new_df10=(new_df10[new_df10.columns[~new_df10.columns.isin(['average','multiplication','summation'])]])
##cpu_scaling_frequency
#
#failed.plot(new_df10)
#failed.plot(new_df10.mean(axis=1))
#failed.set_title(str('CPUscalingfreq10'), fontdict ={'verticalalignment': 'baseline'})
#
#
#fig,failed =  plt.subplots(1,figsize=(16,10))
#df20= pd.read_csv("rochelle20.csv", sep = ";", low_memory=False, error_bad_lines=False)
#df20=mon(df20)
#new_df20=select_group_features(df20,'sipp')
#new_df20=(new_df20[new_df20.columns[~new_df20.columns.isin(['average','multiplication','summation'])]])
##cpu_scaling_frequency
#
#failed.plot(new_df20)
#failed.plot(new_df20.mean(axis=1))
#failed.set_title(str('CPUscalingfreq20'), fontdict ={'verticalalignment': 'baseline'})
#
#
#fig,failed =  plt.subplots(1,figsize=(16,10))
#df40= pd.read_csv("rochelle40.csv", sep = ";", low_memory=False, error_bad_lines=False)
#df20=mon(df20)
#new_df40=select_group_features(df40,'sipp')
#new_df40=(new_df40[new_df40.columns[~new_df40.columns.isin(['average','multiplication','summation'])]])
##cpu_scaling_frequency
#
#failed.plot(new_df40)
#failed.plot(new_df40.mean(axis=1))
#failed.set_title(str('CPUscalingfreq40'), fontdict ={'verticalalignment': 'baseline'})
