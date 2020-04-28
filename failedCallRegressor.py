## For reproducibility
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)




import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import matplotlib.cm as cm
import matplotlib.colors as colors
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import Sequential
from keras.layers import Dense, LSTM 
from sklearn.metrics import confusion_matrix, precision_recall_curve   
from sklearn.model_selection import train_test_split
import re

##import TensorFlow backend
from keras import backend as K
import tensorflow as tf

##GridSearch
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor


##Early stopping
from keras.callbacks import EarlyStopping

##GridSearch
from sklearn.model_selection import GridSearchCV


##Plot keras model
from keras.utils import plot_model

##Dropout
from keras.layers.core import  Dropout



## Flatten a 3D array.
def flatten(X):
    '''    
    Input
    X            A 3D array for lstm, where the array is sample x timesteps x features.
    
    Output
    flattened_X  A 2D array, sample x features.
    '''
    flattened_X = np.empty((X.shape[0], X.shape[2]))  # sample x features array.
    for i in range(X.shape[0]):
        flattened_X[i] = X[i, 0, ]
    return(flattened_X)
    

def smooth3(dfa,interval): #resamples wrt to the specipfed interval making the mean
    timeRange = pd.timedelta_range(start = '0 seconds', end=str((dfa.shape[0]-1)*5)+' seconds',freq='5s')
    dfa['timestampo'] = timeRange
    indexer = pd.TimedeltaIndex(dfa['timestampo'],freq='5s')
    dfa.set_index(indexer,inplace=True)
    dfa_res = dfa.resample(interval).mean()
    dfa_res = dfa_res.select_dtypes(exclude=['object'])
    return dfa_res

###
## Rework metric names in order to have the following pattern:
## <metricname>{<Pod_name>,<CN>}, where CN is POD for pause container and "" for pod-level metrics
###

def reset_Names(columns): 
    #columns = ['container_cpu_load_average_10s{container="",container_name="",id="/kubepods/besteffort/pod1d8d6010-c663-49b3-9000-6483e8d6724c",image="",name="",namespace="openims",pod="pcscf-deployment-84649d986b-mk8jp",pod_name="pcscf-deployment-84649d986b-mk8jp"}','container_cpu_load_average_10s{container="POD",container_name="POD",id="/kubepods/besteffort/pod7700d422-f54b-4754-a2f2-5bb85146bcf5/72d13582b41e2d4a8bf85783c95155b728a7977b1c1046592aac78a01091276a",image="k8s.gcr.io/pause:3.1",name="k8s_POD_pcscf-deployment-7579bc785d-hxsr5_openims_7700d422-f54b-4754-a2f2-5bb85146bcf5_0",namespace="openims",pod="pcscf-deployment-7579bc785d-hxsr5",pod_name="pcscf-deployment-7579bc785d-hxsr5"}','container_cpu_load_average_10s{container="dns",container_name="dns",id="/kubepods/besteffort/pod2be26d89-e5fe-448b-8fbe-8109313b3f47/0f2804be1f1bd7b00b36f2bd369d05eba17efea370aa42f3fc95dca072527019",image="alessiodiama/dns2@sha256:ca80798880cbf2e3f1e38e2f0e308137b5d7566d39e694bc939d4e0c8d6df1f5",name="k8s_dns_dns-deployment-dc7675d4b-b768s_openims_2be26d89-e5fe-448b-8fbe-8109313b3f47_0",namespace="openims",pod="dns-deployment-dc7675d4b-b768s",pod_name="dns-deployment-dc7675d4b-b768s"}','container_network_receive_errors_total{container="POD",container_name="POD",id="/kubepods/besteffort/pod7700d422-f54b-4754-a2f2-5bb85146bcf5/72d13582b41e2d4a8bf85783c95155b728a7977b1c1046592aac78a01091276a",image="k8s.gcr.io/pause:3.1",interface="tunl0",name="k8s_POD_pcscf-deployment-7579bc785d-hxsr5_openims_7700d422-f54b-4754-a2f2-5bb85146bcf5_0",namespace="openims",pod="pcscf-deployment-7579bc785d-hxsr5",pod_name="pcscf-deployment-7579bc785d-hxsr5"}','container_network_receive_packets_dropped_total{container="",container_name="",id="/",image="",interface="cali0ecf3cd1604",name="",namespace="",pod="",pod_name=""}']
    columnsNew = []
    for col in columns:
        colNew = col.split('{')[0]
        #colNew = colNew+'{'+ col.split('pod_name="')[1].split('-')[0]+','+col.split('container=')[1].split(',')[0]+'}'
        if col.split('container=')[1].split(',')[0] != '"POD"' and col.split('container=')[1].split(',')[0] != '""': # to contour errors on pod.yaml
            colNew = colNew+'{'+ col.split('pod_name="')[1].split('-')[0]+','+col.split('pod_name="')[1].split('-')[0]+'}'
        else:
            colNew = colNew+'{'+ col.split('pod_name="')[1].split('-')[0]+','+col.split('container=')[1].split(',')[0]
        ## take in account interface for network related metrics
        if len(col.split('interface="')) != 1:
             colNew = colNew+','+col.split('interface="')[1].split('"')[0]+'}'
        else:
            colNew = colNew+'}'
            
        columnsNew.append(colNew)
    return columnsNew
        
   


type = "Sipp/" #Cadvisor/ Physical/ Sipp
train_dataset = "22_03_Lac102_Full"


if __name__ == '__main__':
       
        ## Load df  
        fileMetrics = [open(type+train_dataset+'/'+file, 'r') for file in os.listdir(type+train_dataset)]
        filesName = []
        pcaArray = []
        prinComponentsArray = []
        sc = MinMaxScaler(feature_range=(0,1))
        means = []
        counter = 0
        for file_handler in fileMetrics: #iterates on toAnalyze files
            filesName.append(file_handler.name.split('/')[1].split('.')[0])
            dfT = pd.DataFrame()
            dfT= pd.read_csv(file_handler, sep = ';', header = 0,low_memory=False, index_col=None, error_bad_lines=False) 
            dfT.drop(list(dfT.filter(regex='StartTime')),axis=1, inplace=True)
            timeRange = pd.timedelta_range(start = '0 minutes', end=str((dfT.shape[0]-1))+' minutes',freq='1min')
            dfT['ApproxFreq'] = timeRange
            indexer = pd.TimedeltaIndex(dfT['ApproxFreq'],freq='1min')
            dfT.set_index(indexer,inplace=True)
            
        indexF = []    
        for u in range(dfT.shape[0]):
            if dfT['FailedCall(P)'][u] != 0:
                indexF.append(u)
                
        plt.figure()
        plt.plot(dfT['FailedCall(P)'][indexF])
        
        ##To be run after reading data in df
        
        
        type = "Cadvisor/" #Cadvisor/ Physical/
        train_dataset = "22_03_Lac102_Full"
        
        plt.rcParams.update({'font.size': 14})
        ## Load file archtypes
        filesType = [open(type+'MetricsTypes/'+file, 'r') for file in os.listdir(type+'MetricsTypes')]
        
        ### per file name-types
        typesFileNames = []
        names = []
        types = []
        ### per file type indexes
        countersIndexes =  []
        gaugesIndexes = []
        untypedIndexes = []
        summaryIndexes = []
        
        for file_handler in filesType: #iterates on folder files
            typesFileNames.append(file_handler.name.split('/')[1].split(".")[0])
            lines = file_handler.readlines()
            #tempMatches = []
            tempTypes = []
            tempNames = []
            for line in lines:
                if len(re.findall('# TYPE',line)) != 0:
                    #print(line)
                    tempTypes.append( line.split('# TYPE ')[1].split(' ')[1].split('\n')[0])
                    tempNames.append(line.split('# TYPE ')[1].split(' ')[0])
            names.append(tempNames)
            types.append(tempTypes)
        
        som = 0
        for u in range(0,len(tempNames)):
            som += len(tempNames[u])
        ## Load df  
        fileMetrics = [open(type+train_dataset+'/'+file, 'r') for file in os.listdir(type+train_dataset)]
        filesName = []
        pcaArray = []
        prinComponentsArray = []
        sc = MinMaxScaler(feature_range=(0,1))
        means = []
        counter = 0
        for file_handler in fileMetrics: #iterates on toAnalyze files
            filesName.append(file_handler.name.split('/')[1].split('.')[0])
            df = pd.DataFrame()
            df= pd.read_csv(file_handler, sep = ';', header = 0,low_memory=False, index_col=None, error_bad_lines=False) 
            ## Common pre-processing
            # Timestamp, Unnamed and obj dropping 
            df = df.select_dtypes(exclude=['object'])
            df.drop(list(df.filter(regex='Unnamed')),axis=1, inplace=True)
            df.drop(list(df.filter(regex='timestamp')),axis=1, inplace=True) #spec start
            df.drop(list(df.filter(regex='spec')),axis=1, inplace=True)
            df.drop(list(df.filter(regex='start')),axis=1, inplace=True) 
            
            df = smooth3(df, '1min') # resampling to match sipp log frequency
            
            
            

        #df.filter(regex='container_memory_rss')
        
            ## Metric types identification
            
            countersTemp = []
            gaugesTemp = []
            untypedTemp = []
            summaryTemp = []
            for j in range(0,len(names[counter])): # for the Cadvisor we retained only "openIms" namespace related metrics
                if types[counter][j] == 'counter':
                    currGropu = df.filter(regex=names[counter][j]+'({+|$)').columns
                    if len(currGropu) != 0 :# to deal with not retained metrics during xml writing after scraping
                        countersTemp.append(currGropu)
                if types[counter][j] == 'gauge':
                    currGropu = df.filter(regex=names[counter][j]+'({+|$)').columns
                    if len(currGropu) != 0 :# to deal with not retained metrics during xml writing after scraping
                        gaugesTemp.append(currGropu)
                if types[counter][j] == 'untyped':
                    currGropu = df.filter(regex=names[counter][j]+'({+|$)').columns
                    if len(currGropu) != 0 :# to deal with not retained metrics during xml writing after scraping
                        untypedTemp.append(currGropu)
                if types[counter][j] == 'summary':
                    currGropu = df.filter(regex=names[counter][j]+'({|$|_sum|_count)').columns
                    if len(currGropu) != 0: # to deal with not retained metrics during xml writing after scraping
                        summaryTemp.append(currGropu)
        

            #drop summary and untyped
            
            for elem in summaryTemp:
                for t in range(0,len(elem)):
                    df.drop(list(df.filter(regex=elem[t])),axis=1, inplace=True)
            for elem in untypedTemp:
                for t in range(0,len(elem)):
                    df.drop(list(df.filter(regex=elem[t])),axis=1, inplace=True)
  
        
            #Counters  pre-processing
            for i in range(len(countersTemp)):
                for j in range(0, len(countersTemp[i])):
                    df[countersTemp[i][j]] = df[countersTemp[i][j]].diff()
     
            for i in range(len(gaugesTemp)): # tries to catch wrongly marked gauges
                for j in range(0, len(gaugesTemp[i])):
                    if  '_total' in gaugesTemp[i][j]:
                        df[gaugesTemp[i][j]] = df[gaugesTemp[i][j]].diff()
                        #print(gaugesTemp[i][j])
            
            
            df=df.fillna(value=0) # first elem will be Nan after diff
        
        
        
        dfCut = df.iloc[0:dfT.shape[0],:]
        joinedDf = dfCut.join(dfT)
        
        reverse_df =  joinedDf.iloc[::-1] # reverse df for lookback
        
               
     
        # Gauges pre-processing
        ## nothing to do
        
        #to drop container_memory_max_usage_bytes
        
        #TODO add tempearture
        # REGEX filtering
        regExString = 'cpu|network' #  _cpu_|_network_     pod_name="dns-deployment-6d486c4fb6-slsq6"     .* -> all features, change for use specific group of features
#        regEXX = "pcscf"
        columnToUse = reverse_df.filter(regex=regExString) 
#        columnToUse = columnToUse.filter(regex=regEXX) 
        
        
        oderedColumn = reset_Names(columnToUse.columns)
        oderedColumn = [el.replace('"','') for el in oderedColumn]
        toDelete = list( el for el in oderedColumn if 'cpu' in el and ( 'POD' in el or ',}' in el or ',,' in el) )
        columnToUse.columns = oderedColumn
        columnToUse.drop(toDelete,axis=1, inplace=True)
        
#        columnToUse.drop(list(columnToUse.filter(regex='container_memory_max')),axis=1, inplace=True)
        #columnToUse[list(columnToUse.filter(regex='.*_bytes|memory'))] /= 1000000000
        
        feature_size = len(columnToUse.columns)
        input_feature_resampeled_normalized  = columnToUse.values
        
        
        input_feature_resampeled_normalized = sc.fit_transform(input_feature_resampeled_normalized)
        #back = sc.inverse_transform(input_feature_resampeled_normalized2)
        ## LSTM data format and rescaling
        dt_temp = pd.DataFrame(input_feature_resampeled_normalized, columns=columnToUse.columns)
   
        ## split into test and validation
        
#            SEED = 123 #used to help randomly select the data points
#            DATA_SPLIT_PCT = 0.2
#            # testo are validation data
#            input_data_train, input_data_testo = train_test_split(input_feature_resampeled_normalized,test_size=DATA_SPLIT_PCT, shuffle=False)
#            #(input_data_train = input_data
#            input_data_train_df = pd.DataFrame(input_data_train,columns=columnToUse.columns)
        
        input_data_train = input_feature_resampeled_normalized
        #lookback
    
        lookback = 4
        
        ##LSTM format training
        X_train_A_look=[]
        for i in range(len(input_data_train)-lookback-1) :
            t=[]
            for j in range(0,lookback):
                t.append(input_data_train[[(i+j)], :])
            X_train_A_look.append(t)
        
        X= np.array(X_train_A_look)
      
        X_train = X.reshape(X.shape[0],lookback, feature_size)
        X_train = np.flip(X_train,0)
        
        y_train = reverse_df.filter(regex='FailedCall\(P\)').values
        ##LSTM format training
        y_train_A_look=[]
        for i in range(len(y_train)-lookback-1) :
            t=[]
            for j in range(0,lookback):
                t.append(y_train[[(i+j)]])
            y_train_A_look.append(t)
        
        Y= np.array(y_train_A_look)
      
        Y_train = Y.reshape(Y.shape[0],lookback, 1)
        Y_train = np.flip(Y_train,0)
        
     
        callbacks_list = []
        ##TensorBoard to monitor
        tb = TensorBoard(log_dir='.summary_dir',
                        histogram_freq=0,
                        write_graph=True,
                        write_images=True)
        
        
        def create_model(feature_size):
            hidden_layer_size = int(feature_size*0.8)
            hidden_layer_size2 = int(feature_size*0.6)
            hidden_layer_size3 = int(feature_size*0.4)
            lstm_autoencoder = Sequential()
            # Encoder
            lstm_autoencoder.add(LSTM(hidden_layer_size, activation='relu', input_shape=(lookback,feature_size), return_sequences=True, name = 'encode1'))
            lstm_autoencoder.add(Dropout(0.2, name = 'dropout1'))
            lstm_autoencoder.add(LSTM(hidden_layer_size2, activation='relu', return_sequences=True, name = 'encode2'))
            lstm_autoencoder.add(Dropout(0.2, name = 'dropout2'))
            lstm_autoencoder.add(LSTM(hidden_layer_size3, activation='relu', return_sequences=True, name = 'encode3'))
            lstm_autoencoder.add(Dropout(0.2, name = 'dropout3'))
            lstm_autoencoder.add(Dense(units=1))
            lstm_autoencoder.summary()         
        
            lstm_autoencoder.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])
            return lstm_autoencoder
        
        
        
        
        
        
        #KerasClassifier
    
        lstm_autoencoder = create_model(feature_size)
#        # save model to single file
#        model.save('lstm_model.h5')
        plot_model(lstm_autoencoder, show_shapes=True,to_file='reconstruct_lstm_autoencoder.png')
        
        ##checkpoint to monitor training
        filepath="weights-improvement-{epoch:02d}-{loss:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=10, save_best_only=True, mode='min')
        ##Early stopping
        es = EarlyStopping(monitor='loss',patience=10, mode='min')
        
        callbacks_list.append(checkpoint)
        callbacks_list.append(tb)
        callbacks_list.append(es)
        
        
    
        #sinlge train
        lstm_autoencoder_history = lstm_autoencoder.fit(X_train, Y_train, epochs=500, batch_size=16,verbose=2,callbacks=callbacks_list).history
    
#        plt.figure()
#        #plt.plot(lstm_autoencoder_history['loss'], linewidth=2, label='Train')
#        plt.plot(lstm_autoencoder_history['acc'], linewidth=2, label='Valid')
#        plt.legend(loc='upper right')
#        plt.title('Model loss')
#        plt.ylabel('Acc')
#        plt.xlabel('Epoch')
#        plt.show()
        

    
    ## Learn reconstruction error on training to decide anomalt treshold
        
    
        #prediction_train = modelNew.predict(X_train)
        prediction_train = lstm_autoencoder.predict(X_train) #  Train case
        
        
        reconstructed_train_retrans = flatten(prediction_train)
        reconstructed_train_retrans =  reconstructed_train_retrans
        train_retrans = (flatten(Y_train))
#        train_retrans = train_retrans.round()
        #prediction_val = lstm_autoencoder.predict(X_testo) #  Train case
        #reconstructed_retrans = sc.inverse_transform(flatten(prediction_train))
        
        plt.figure()
        plt.plot(train_retrans)
        plt.title("Train "+train_dataset)
        plt.figure()
        plt.plot(reconstructed_train_retrans)
        plt.title("Reconstructed Train "+train_dataset)
        
        RMSE =  np.power(np.mean(np.power(train_retrans-reconstructed_train_retrans,2),axis=1),1/2)
        plt.figure()
        plt.plot(RMSE)
        
        
        
        
        
#        type = "Cadvisor/" #Cadvisor/ Physical/        
#        plt.rcParams.update({'font.size': 14})
#        ## Load file archtypes
#        filesType = [open(type+'MetricsTypes2/'+file, 'r') for file in os.listdir(type+'MetricsTypes2')]
#        
#        ### per file name-types
#        typesFileNames = []
#        names = []
#        types = []
#        ### per file type indexes
#        countersIndexes =  []
#        gaugesIndexes = []
#        untypedIndexes = []
#        summaryIndexes = []
#        
#        for file_handler in filesType: #iterates on folder files
#            typesFileNames.append(file_handler.name.split('/')[1].split(".")[0])
#            lines = file_handler.readlines()
#            #tempMatches = []
#            tempTypes = []
#            tempNames = []
#            for line in lines:
#                if len(re.findall('# TYPE',line)) != 0:
#                    #print(line)
#                    tempTypes.append( line.split('# TYPE ')[1].split(' ')[1].split('\n')[0])
#                    tempNames.append(line.split('# TYPE ')[1].split(' ')[0])
#            names.append(tempNames)
#            types.append(tempTypes)
        
        
        predicted_values = []
        input_data_array = []
        type = "Cadvisor/" #Cadvisor/ Physical/
        test_dataset = '20_03' #StressScscf30min
        files2 = [open(type+test_dataset+"/"+file, 'r') for file in os.listdir(type+test_dataset)]
        for file_handler2 in files2: #iterates on toAnalyze files
            df2 = pd.DataFrame()
            df2 = pd.read_csv(file_handler2, sep = ";", header = 0,low_memory=False, index_col=None, error_bad_lines=False)
            df2 = df2.select_dtypes(exclude=['object'])
            df2.drop(list(df2.filter(regex='Unnamed')),axis=1, inplace=True)
            df2.drop(list(df2.filter(regex='timestamp')),axis=1, inplace=True) #spec start
            df2.drop(list(df2.filter(regex='spec')),axis=1, inplace=True)
            df2.drop(list(df2.filter(regex='start')),axis=1, inplace=True)
             
            df2 = smooth3(df2, '60s')
#            df2 = df2.iloc[1440,:] # cut off to
            
            countersTemp = []
            gaugesTemp = []
            untypedTemp = []
            summaryTemp = []
            for j in range(0,len(names[counter])): # for the Cadvisor we retained only "openIms" namespace related metrics
                if types[counter][j] == 'counter':
                    currGropu = df2.filter(regex=names[counter][j]+'({+|$)').columns
                    if len(currGropu) != 0 :# to deal with not retained metrics during xml writing after scraping
                        countersTemp.append(currGropu)
                if types[counter][j] == 'gauge':
                    currGropu = df2.filter(regex=names[counter][j]+'({+|$)').columns
                    if len(currGropu) != 0 :# to deal with not retained metrics during xml writing after scraping
                        gaugesTemp.append(currGropu)
                if types[counter][j] == 'untyped':
                    currGropu = df2.filter(regex=names[counter][j]+'({+|$)').columns
                    if len(currGropu) != 0 :# to deal with not retained metrics during xml writing after scraping
                        untypedTemp.append(currGropu)
                if types[counter][j] == 'summary':
                    currGropu = df2.filter(regex=names[counter][j]+'({|$|_sum|_count)').columns
                    if len(currGropu) != 0: # to deal with not retained metrics during xml writing after scraping
                        summaryTemp.append(currGropu)
            
           
            
            #drop summary and untyped
            
            for elem in summaryTemp:
                for t in range(0,len(elem)):
                    df2.drop(list(df2.filter(regex=elem[t])),axis=1, inplace=True)
            for elem in untypedTemp:
                for t in range(0,len(elem)):
                    df2.drop(list(df2.filter(regex=elem[t])),axis=1, inplace=True)
  
          
            #Counters  pre-processing
            for i in range(len(countersTemp)):
                for j in range(0, len(countersTemp[i])):
                    df2[countersTemp[i][j]] = df2[countersTemp[i][j]].diff()
     
            for i in range(len(gaugesTemp)): # tries to catch wrongly marked gauges
                for j in range(0, len(gaugesTemp[i])):
                    if  '_total' in gaugesTemp[i][j]:
                        df2[gaugesTemp[i][j]] = df2[gaugesTemp[i][j]].diff()
                        #print(gaugesTemp[i][j])
            
            
            df2=df2.fillna(value=0) # first elem will be Nan after diff
            reverse_df2 =  df2.iloc[::-1] # reverse df for lookback
        
            # Gauges pre-processing
            ## nothing to do
                
            # Normalization
            #regExString = 'cpu|network' #  _cpu_|_network_     pod_name="dns-deployment-6d486c4fb6-slsq6"     .* -> all features, change for use specific group of features
            columnToUse2 = reverse_df2.filter(regex=regExString)  
            cols2 = reset_Names(columnToUse2.columns)
#            oderedColumn = oderedColumn[:1]
            cols2 = [el.replace('"','') for el in cols2]
            columnToUse2.columns = cols2
            columnToUse2 = columnToUse2[oderedColumn]
            
            
            for e in cols2:
                if e not in oderedColumn:
                    print("aaaaa")
        
#            sc2 = MinMaxScaler(feature_range=(0,1))
          
            input_feature_rescaled_normalized = sc.transform(columnToUse2)
            
            dt_temp2 = pd.DataFrame(input_feature_rescaled_normalized,columns=columnToUse2.columns)

#    
#    
#             indexes = []
#             boolA  = dt_temp2.mean() > 1  
#             for o in range(0,len(boolA)):
#                if boolA[o] == True:
#                    print(o)    
#                    indexes.append(o)
#            colls = dt_temp2.iloc[:,indexes].columns
                                
            ## LSTM data format preparation
        
            X2=[]
            ##LSTM format training
            X_test_A_look=[]
            for i in range(len(input_feature_rescaled_normalized)-lookback) :
                t=[]
                for j in range(0,lookback):
                    t.append(input_feature_rescaled_normalized[[(i+j)], :])
                X_test_A_look.append(t)
            
            X= np.array(X_test_A_look)
          
            X_test = X.reshape(X.shape[0],lookback, feature_size)
            X_test = np.flip(X_test,0)
           
            #prediction_test = modelNew.predict(X_test) #  Test case
            prediction_test = lstm_autoencoder.predict(X_test) #  Test case
            
            
            reconstructed_test_retrans = (flatten(prediction_test))
#            test_retrans = sc.inverse_transform(flatten(X_test))
            
            plt.figure()
            plt.plot(reconstructed_test_retrans)
            plt.title("reconstructed "+test_dataset)
#            plt.figure()
#            plt.plot(test_retrans)  
#            plt.title("Input Test "+test_dataset)
        