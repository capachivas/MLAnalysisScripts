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


class ErrorClass:
   def __init__(self, pattern,timeIndex):
        self.pattern = pattern
#        self.indexes = indexes
        self.timeIndex = timeIndex
        self.containers = []
        self.freq = 0
    
   def set_containers(self,containers):
       self.containers.append(containers)
       
   def set_freq(self,freq):
       self.freq = freq




podsRegEx ={'pod_name="dns-deployment-6d486c4fb6-slsq6"': 'dns',
            'pod_name="scscf-deployment-56679d845-cnjr7"': 'scscf',
            'pod_name="hss-deployment-7b9c7786fd-qgvts"': 'hss',
            'pod_name="icscf-deployment-74f5bb8559-wppfp"': 'icscf',
            'pod_name="pcscf-deployment-84649d986b-mk8jp"': 'pcscf'}
        
#        } ['pod_name="dns-deployment-6d486c4fb6-slsq6"',
#'pod_name="scscf-deployment-56679d845-cnjr7"',
#'pod_name="hss-deployment-7b9c7786fd-qgvts"',
#'pod_name="icscf-deployment-74f5bb8559-wppfp"',
#'pod_name="pcscf-deployment-84649d986b-mk8jp"']

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
    

def smooth3(dfa,interval): #'60s'
    timeRange = pd.timedelta_range(start = '0 seconds', end=str((dfa.shape[0]-1)*5)+' seconds',freq='5s')
    dfa['timestampo'] = timeRange
    indexer = pd.TimedeltaIndex(dfa['timestampo'],freq='5s')
    dfa.set_index(indexer,inplace=True)
    dfa_res = dfa.resample(interval).mean()
    dfa_res = dfa_res.select_dtypes(exclude=['object'])
    return dfa_res
   


type = "Cadvisor/" #Cadvisor/ Physical/
train_datasets = ["16_03","17_03","18_03","19_03"]
df_arr = []

if __name__ == '__main__':
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


    ## Load dfs
    for train_dataset in train_datasets:
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
            
            
            df.filter(regex='container_memory_working_set_bytes')
            
            df = smooth3(df, '30s') # resampling to match sipp log frequency
        
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
            reverse_df =  df.iloc[::-1] # reverse df for lookback
            df_arr.append(reverse_df)

    concat_df = pd.concat(df_arr,ignore_index=True,sort=False)
 
  
    
    #TODO add tempearture
    # REGEX filtering
    regExString = 'cpu' #  _cpu_|_network_     pod_name="dns-deployment-6d486c4fb6-slsq6"     .* -> all features, change for use specific group of features
    columnToUse = concat_df.filter(regex=regExString)  
    columnToUse.drop(list(columnToUse.filter(regex='container_memory_max')),axis=1, inplace=True)
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
    
    input_data_train = input_feature_resampeled_normalized#[len(input_feature_resampeled_normalized)-1000:len(input_feature_resampeled_normalized)]
    #lookback
    
    lookback = 2
    
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
     
    callbacks_list = []
    ##TensorBoard to monitor
    tb = TensorBoard(log_dir='.summary_dir',
                    histogram_freq=0,
                    write_graph=True,
                    write_images=True)
    
    
    def create_model(feature_size):
        hidden_layer_size = int(feature_size*0.8)
        print(hidden_layer_size)
        hidden_layer_size2 = int(feature_size*0.6)
        print(hidden_layer_size2)
        lstm_autoencoder = Sequential()
        # Encoder
        lstm_autoencoder.add(LSTM(hidden_layer_size, activation='elu', input_shape=(lookback,feature_size), return_sequences=True, name = 'encode1'))
        lstm_autoencoder.add(Dropout(0.2, name = 'dropout_encode_1'))
        lstm_autoencoder.add(LSTM(hidden_layer_size2, activation='elu', return_sequences=False, name = 'encode2'))
        lstm_autoencoder.add(RepeatVector(lookback))
        lstm_autoencoder.add(LSTM(hidden_layer_size, activation='elu', return_sequences=True, name = 'dencode1'))
        lstm_autoencoder.add(Dropout(0.2, name = 'dropout_dencode_1'))
        lstm_autoencoder.add(LSTM(feature_size, activation='linear', return_sequences=True, name = 'dencode2'))
      
    
        lstm_autoencoder.summary()         
    
        lstm_autoencoder.compile(optimizer='adam', loss='mean_squared_error',metrics=['mse'])
        return lstm_autoencoder
    
    
    
    
    

    #KerasClassifier
    
    lstm_autoencoder = create_model(feature_size)
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
#    lstm_autoencoder_history = lstm_autoencoder.fit(X_train, X_train, epochs=500, batch_size=32,verbose=2,callbacks=callbacks_list).history
#    lstm_autoencoder_history = lstm_autoencoder.fit(X_train, X_train, epochs=500, batch_size=256,verbose=2,callbacks=callbacks_list).history
    
    modelNew = create_model(feature_size)
    modelNew.load_weights('weights-improvement-414-0.00.hdf5')
        
    
    
    #prediction_train = lstm_autoencoder.predict(X_train) #  Train case
    prediction_train = modelNew.predict(X_train)
    
    
    reconstructed_train_retrans = sc.inverse_transform(flatten(prediction_train))
    train_retrans = sc.inverse_transform(flatten(X_train))
    #prediction_val = lstm_autoencoder.predict(X_testo) #  Train case
    #reconstructed_retrans = sc.inverse_transform(flatten(prediction_train))
    
    plt.figure()
    plt.plot(train_retrans)
    plt.title("Train "+train_dataset)
    plt.figure()
    plt.plot(reconstructed_train_retrans)
    plt.title("Reconstructed Train "+train_dataset)
    
    scored = pd.DataFrame(index = range(0,X_train.shape[0]))
    scored['Loss_mae'] = np.mean(np.power(train_retrans-reconstructed_train_retrans,2), axis=1)
   
    #col_mean =  np.mean(np.power(prediction_train-input_feature_df,2), axis=0)
    plt.figure(figsize=(16,9), dpi=80)
  
    sns.distplot(scored['Loss_mae'],bins = 2000, kde = True, color = 'blue',norm_hist=True)
    plt.title('Train error distribution')
    thresholds = [ scored['Loss_mae'].mean()+scored['Loss_mae'].std(),  scored['Loss_mae'].mean()+2*scored['Loss_mae'].std(),  scored['Loss_mae'].mean()+3*scored['Loss_mae'].std()]
    scored['Threshold1'] = thresholds[0]
    scored['Anomaly1'] = scored['Loss_mae'] > scored ['Threshold1']
    scored['Threshold2'] =  thresholds[1]
    scored['Anomaly2'] = scored['Loss_mae'] > scored ['Threshold2']
    scored['Threshold3'] =  thresholds[2]
    scored['Anomaly3'] = scored['Loss_mae'] > scored ['Threshold3']
    #scored.head()
    scored.plot(title='Error Train '+train_dataset)
    
    errorDfSquare = pd.DataFrame(np.power(train_retrans-reconstructed_train_retrans,2),columns = columnToUse.columns)
    
    # T1 : scored.iloc[q,2] and not scored.iloc[q,4]  and  not scored.iloc[q,6]
    # T2 : scored.iloc[q,2] and scored.iloc[q,4]  and  not scored.iloc[q,6]
    # T3 : scored.iloc[q,6]
    
    
    perEccr = [] # perEccr[i] list of %contribution to MSI for i-th anomaly
    timeIndexes = []# time index of most contributing error feature
    errIndexes = [] # sorted index of most error contirubuting feature
    errorSums = []
    for q in range(0,scored.shape[0]): #scored.shape[0]
        if scored.iloc[q,6] : 
            timeIndexes.append(q)
            perEccrTemp = []
            sumErr = (errorDfSquare.iloc[q,:].sum())
            perEccrTemp = [errQ/sumErr for errQ in errorDfSquare.iloc[q,:]]
            perEccr.append(perEccrTemp)
#            print(perEccrTemp)
            
#            plt.figure()
#            plt.bar(np.linspace(1,feature_size,feature_size),perEccrTemp)
            
            perEccrTempSorted = np.flip(np.sort(np.asanyarray(perEccrTemp))) #sort ascending % error
            tempSum = 0
            stopIndex = 0
            for w in range(0,len(perEccrTempSorted)): # serah first stopIndex features that reaches 0.8% of total error
                tempSum += perEccrTempSorted[w]
                if tempSum >= 0.8:
                    stopIndex = w
                    errorSums.append(tempSum)
                    break
            errIndexesTemp = []
            for j in range(0,stopIndex+1):
                errIndexesTemp.append(np.where(perEccrTemp == perEccrTempSorted[j])[0][0]) # look for index of the j-th perEccrTempSorted in perEccrTemp
            errIndexes.append(errIndexesTemp)
    
    #TODO take into account permutations
    # Some errors are charaterized by permutation of same metrics, that is 
    # the same metrics charaterize the deviation but the magnitude of the influence of each metric is different
    # Considering permutation we just cahtch type of deviation, negleting each metric contribution
    # Further study ordered contribution will further charatherize the deviation
    
    
    # Store pattern wise deviation classes putting togheter 
    # errorClasses will contain a pattern and the list of timestamps where this pattern happens
    
    errorClasses = [] 
    for w in range(0,len(errIndexes)):
        currClass = errIndexes[w]
        if currClass not in [c.pattern for c in errorClasses]: # if it is a not known error class
            indexes = [] #collect indexes of this pattern class
            indexes.append(w)
            for q in range(w+1,len(errIndexes)):
                if currClass == errIndexes[q]:
                    indexes.append(q)
            ia = np.asarray([el for el in indexes],dtype=int)
            errorClasses.append(ErrorClass(currClass,np.array(timeIndexes)[ia]))


    # here we take care of permutations
    
    from sympy.utilities.iterables import multiset_permutations
    
    scannedClasses = []
    usedIndex = [] #root Classes
    permUsedIndex = [] # permUsedIndex[i] := list of permutations of usedIndex[i] root class
    for w in range(0,len(errorClasses)):
        currClass = errorClasses[w].pattern
        if currClass not in scannedClasses:
            usedIndex.append(w)
            scannedClasses.append(currClass)
            permutations = multiset_permutations(currClass)
            lista = []     
            for e in permutations:
                lista.append(e)
            permIndex = []
            for q in range(w+1,len(errorClasses)):
                if errorClasses[q].pattern in lista: # faund a permutation
                    permIndex.append(q)
                    scannedClasses.append(errorClasses[q].pattern)
            permUsedIndex.append(permIndex)
   
    cleanClasses = [] #cleanClasses[i] contains root class and all its permutations
    
    for l in range(0,len(permUsedIndex)):
        if len(permUsedIndex[l]) == 0:
            cleanClasses.append(errorClasses[usedIndex[l]])  
        else:
            pattern = errorClasses[usedIndex[l]].pattern
            indexesN = errorClasses[usedIndex[l]].timeIndex
            for y in range (len(permUsedIndex[l])):
                indexesN = np.append(indexesN,errorClasses[permUsedIndex[l][y]].timeIndex)
            newErrClass = ErrorClass(pattern,indexesN)
            cleanClasses.append(newErrClass)  
            
            
            
    # get patterns and timestamps        
    [c.pattern for c in cleanClasses]    
    [c.timeIndex for c in cleanClasses]  

    [c.pattern for c in errorClasses]    
    [c.timeIndex for c in errorClasses]     

    ###
    ## Group frequency
    ###        
      
    groupFrequency = []
    for i in range(len(cleanClasses)):
        cleanClasses[i].set_freq(len(cleanClasses[i].timeIndex)/len(timeIndexes)*100)
    
    ###
    ## Get per timestamp feature percentage to MSE(t)
    ###
    
   
    
    def perc_class(errClass):
        errClass=cleanClasses[0]
        percMSE = []
        for t in range(len(errClass.timeIndex)):
            numAnomlay = timeIndexes.index(errClass.timeIndex[t])
            percMSETemp = []
            for l in range(len(errClass.pattern)):
                percMSETemp.append(perEccr[numAnomlay][errClass.pattern[l]])
            percMSE.append(percMSETemp)
        return percMSE
    
    
      print(perc_class(cleanClasses[0]))      
    

    # Extract feature container and type
    # Todo: optimize as it perform the same job for the same metric severla times       
    metDictL = []   
    for el in cleanClasses:
        metNames =  errorDfSquare.iloc[:,el.pattern].columns
        metDict = []  
        for met in metNames:
            metType = met.split(',pod="')[1].split('"')[0]
            container = met.split('{')[0]
            metDict.append(metType) 
            metDict.append(container) 
        metDictL.append(metDict)
            
            
            
            
            
            
###Test Phase            
    
    predicted_values = []
    input_data_array = []
    test_dataset = '16_03+Stress' #StressScscf30min
    files2 = [open(type+test_dataset+"/"+file, 'r') for file in os.listdir(type+test_dataset)]
    for file_handler2 in files2: #iterates on toAnalyze files
        df2 = pd.DataFrame()
        df2 = pd.read_csv(file_handler2, sep = ";", header = 0,low_memory=False, index_col=None, error_bad_lines=False)
        df2 = df2.select_dtypes(exclude=['object'])
        df2.drop(list(df2.filter(regex='Unnamed')),axis=1, inplace=True)
        df2.drop(list(df2.filter(regex='timestamp')),axis=1, inplace=True)
        #df2.drop(list(df.filter(regex='memory')),axis=1, inplace=True)
         
        df2 = smooth3(df2, '20s')
        
       
        
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
        columnToUse2 = reverse_df2.filter(columnToUse.columns) #use same columns as in training phase
        #columnToUse2[list(columnToUse2.filter(regex='.*_bytes|memory'))] /= 1000000000 #I rescaling everithing is in byte and memory related stats

    
        sc2 = MinMaxScaler(feature_range=(0,1))
      
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
       
        prediction_test = modelNew.predict(X_test) #  Test case
#        prediction_test = lstm_autoencoder.predict(X_test) #  Test case
       
        reconstructed_test_retrans = sc.inverse_transform(flatten(prediction_test))
        test_retrans = sc.inverse_transform(flatten(X_test))
        
        plt.figure()
        plt.plot(reconstructed_test_retrans)
        plt.title("reconstructed "+test_dataset)
        plt.figure()
        plt.plot(test_retrans)  
        plt.title("Input Test "+test_dataset)
        
        
#            perFeatureMSE2 = np.mean(sc.inverse_transform(flatten(X_test))-reconstructed_test_retrans,2), axis=0)
#            plt.figure()
#            plt.plot(perFeatureMSE2)
#        
        
        
#            errorArray2 = sc.inverse_transform(flatten(X_test))-reconstructed_test_retrans
#            errorDf = pd.DataFrame(errorArray2, columns = columnToUse.columns)
#            
#            x_surface = []
#            y_surface = []
#            z_surface = []
#            for x in range(1,feature_size+1):
#                for y in range(1,int((errorArray2.shape[0]+1)/4)):
#                    x_surface.append(x)
#                    y_surface.append(y)
#                    z_surface.append((errorArray2[y-1][x-1]))
#            
#            fig = plt.figure()
#            ax = fig.gca(projection='3d')
#            
#            ax.plot_trisurf(x_surface, y_surface, z_surface, linewidth=0.2, antialiased=True)
#            

        scored2 = pd.DataFrame(index = range(0,X_test.shape[0]))
        scored2['Loss_mae'] = np.mean(np.power(test_retrans- reconstructed_test_retrans,2), axis=1)
#            plt.figure(figsize=(16,9), dpi=80)
        #sns.distplot(scored2['Loss_mae'],bins = 20000, kde = True, color = 'blue',norm_hist=True)
        #thresholds = [ scored['Loss_mae'].mean()+scored['Loss_mae'].std(),  scored['Loss_mae'].mean()+2*scored['Loss_mae'].std(),  scored['Loss_mae'].mean()+3*scored['Loss_mae'].std()]
        scored2['Threshold1'] = thresholds[0]
        scored2['Anomaly1'] = scored2['Loss_mae'] > scored2 ['Threshold1']
        scored2['Threshold2'] =  thresholds[1]
        scored2['Anomaly2'] = scored2['Loss_mae'] > scored2 ['Threshold2']
        scored2['Threshold3'] =  thresholds[2]
        scored2['Anomaly3'] = scored2['Loss_mae'] > scored2 ['Threshold3']
        #scored.head()
        scored2.plot(title='Error Test '+test_dataset)
        
        
        
  # only in class T1 ->   scored2.iloc[q,2] == True and scored2.iloc[q,4] == False and scored2.iloc[q,6] == False  
  # only in class T2 ->   scored2.iloc[q,4] == True and scored2.iloc[q,6] == False  
  # only in class T3 ->   scored2.iloc[q,6] == True
        
        
        
    errorDfSquare = pd.DataFrame(np.power(test_retrans- reconstructed_test_retrans,2),columns = columnToUse.columns)
    
    perEccr = []
    timeIndexes = []
    errIndexes = []
    errorSums = []
    for q in range(0,scored2.shape[0]): #scored.shape[0]
        if scored2.iloc[q,6] == True :
            timeIndexes.append(q)
            perEccrTemp = []
            sumErr = np.mean(errorDfSquare.iloc[q,:].sum())
            perEccrTemp = [errQ/sumErr for errQ in errorDfSquare.iloc[q,:]]
            perEccr.append(perEccrTemp)
#            print(perEccrTemp)
            
#            plt.figure()
#            plt.bar(np.linspace(1,feature_size,feature_size),perEccrTemp)
            
            perEccrTempSorted = np.flip(np.sort(np.asanyarray(perEccrTemp)))
            tempSum = 0
            stopIndex = 0
            for w in range(0,len(perEccrTempSorted)):
                tempSum += perEccrTempSorted[w]
                if tempSum >= 0.9:
                    stopIndex = w
                    errorSums.append(tempSum)
                    break
            errIndexesTemp = []
            for j in range(0,stopIndex+1):
                errIndexesTemp.append(np.where(perEccrTemp == perEccrTempSorted[j])[0][0])
            errIndexes.append(errIndexesTemp)
    
    #TODO take into account permutations
#    times = -1
    errorClasses = []
    for w in range(0,len(errIndexes)):
        currClass = errIndexes[w]
        if currClass not in [c.pattern for c in errorClasses]: # if it is a not known error class
    #        times += 1
            indexes = []
            indexes.append(w)
            for q in range(w+1,len(errIndexes)):
                if currClass == errIndexes[q]:
                    indexes.append(q)
#            timeIndexes = [c.indexes for c in errorClasses]
            ia = np.asarray([el for el in indexes],dtype=int)
            errorClasses.append(ErrorClass(currClass,np.array(timeIndexes)[ia]))

    from sympy.utilities.iterables import multiset_permutations
    
    scannedClasses = []
    usedIndex = []
    permUsedIndex = []
    for w in range(0,len(errorClasses)):
        currClass = errorClasses[w].pattern
        if currClass not in scannedClasses:
            usedIndex.append(w)
            scannedClasses.append(currClass)
            permutations = multiset_permutations(currClass)
            lista = []     
            for e in permutations:
                lista.append(e)
            permIndex = []
            for q in range(w+1,len(errorClasses)):
                if errorClasses[q].pattern in lista: # faund a permutation
                    permIndex.append(q)
                    scannedClasses.append(errorClasses[q].pattern)
            permUsedIndex.append(permIndex)
   
    cleanClasses = [] #classes without permutations of patterns
    
    for l in range(0,len(permUsedIndex)):
        if len(permUsedIndex[l]) == 0:
            cleanClasses.append(errorClasses[usedIndex[l]])  
        else:
            pattern = errorClasses[usedIndex[l]].pattern
            indexesN = errorClasses[usedIndex[l]].timeIndex
            for y in range (len(permUsedIndex[l])):
                indexesN = np.append(indexesN,errorClasses[permUsedIndex[l][y]].timeIndex)
            newErrClass = ErrorClass(pattern,indexesN)
            cleanClasses.append(newErrClass)  
            
            
            
    # get patterns and timestamps    

    [c.pattern for c in cleanClasses]    
    [c.timeIndex for c in cleanClasses]       
    
    [c.pattern for c in errorClasses]    
    [c.timeIndex for c in errorClasses]


   ###
    ## Group frequency
    ###        
      
    groupFrequency = []
    for i in range(len(cleanClasses)):
        groupFrequency.append(len(cleanClasses[i].timeIndex)/len(timeIndexes)*100)
           
              
    metDictL = []   
    for el in cleanClasses:
        metNames =  errorDfSquare.iloc[:,el.pattern].columns
        metDict = []  
        for met in metNames:
            metType = met.split(',pod="')[1].split('"')[0]
            container = met.split('{')[0]
            metDict.append(metType) 
            metDict.append(container) 
        metDictL.append(metDict)
        
        
    
    
    
    
    