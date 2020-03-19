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
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import Sequential
from keras.layers import Dense, LSTM 
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.preprocessing import RobustScaler    
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
        flattened_X[i] = X[i, (X.shape[1]-1), ]
    return(flattened_X)
    
def smooth(X, colToSmooth): #takes in input an array [rows][col] where col:==features and rows==:timestamps and smooths it
###Smoothing
    ## Speeds up computation not smoothing stable variables   
    X = X.transpose()
    halfWindow = 6 #TODO 1 min
    X_smooth = []
    for  col in range(0,X.shape[0]): # colToSmooth: 
        X_smooth_temp = []
        countInf = 0
        countSup = halfWindow
        if col in colToSmooth:
            for center in range(0,X.shape[1]):#range(halfWindow,X_test.shape[1]-halfWindow):
                if center == 0 :
                    X_smooth_temp.append(X[col][0])
                    countInf += 1
                elif center < halfWindow and center != 0 :
                    X_smooth_temp.append(np.mean(X[col][0:center+countInf+1]))
                    countInf += 1
                elif center == X.shape[1]-1:
                     X_smooth_temp.append(X[col][X.shape[1]-1])
                elif center >= X.shape[1]-halfWindow:
                    X_smooth_temp.append(np.mean(X[col][center-countSup:X.shape[1]-1+1]))
                    countSup -= 1
                else :
                    X_smooth_temp.append(np.mean(X[col][center-halfWindow:center+halfWindow+1]))
            X_smooth.append(X_smooth_temp)
        else:
            X_smooth.append(X[col])
        X_A = np.array(X_smooth)
        X_A = X_A.transpose()
    return X_A



def smooth2(X, n): #drops n timestamps each n+1 timestamps on X
    return  X.iloc[::n]


type = "Cadvisor/" #Cadvisor/ Physical/
train_dataset = "baseline"


if __name__ == '__main__':
#     with tf.Session(config=tf.ConfigProto(
#                            intra_op_parallelism_threads=16)) as sess:
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
        sc= StandardScaler() ## we cannot use minMax as we do not know min and max of future data
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
            df.drop(list(df.filter(regex='timestamp')),axis=1, inplace=True)
            #df.drop(list(df.filter(regex='memory')),axis=1, inplace=True)
            #df.drop(list(df.filter(regex='inodes')),axis=1, inplace=True)
#            df.drop(list(df.filter(regex='container_fs_usage_bytes')),axis=1, inplace=True)
#            df.drop(list(df.filter(regex='container_last_seen')),axis=1, inplace=True)
            
            
            
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
  
            
            #df = df.iloc[0:int(df.shape[0]/26),]
        
            #Counters  pre-processing
            for i in range(len(countersTemp)):
                for j in range(0, len(countersTemp[i])):
                    df[countersTemp[i][j]] = df[countersTemp[i][j]].diff()
     
                     
            df=df.fillna(value=0) # first elem will be Nan after diff
            dfPerc = df.pct_change(periods=600)
            reverse_df =  df.iloc[::-1] # reverse df for lookback
            
            # Gauges pre-processing
            ## nothing to do
            
            
            # REGEX filtering
            regExString = 'scscf' # .* -> all features, change for use specific group of features
            columnToUse = reverse_df.filter(regex=regExString)  
            columnToUse.drop(list(columnToUse.filter(regex='container_memory_cache')),axis=1, inplace=True)
            columnToUse.drop(list(columnToUse.filter(regex='container_spec')),axis=1, inplace=True)
            columnToUse.drop(list(columnToUse.filter(regex='container_memory_max')),axis=1, inplace=True)
            columnToUse.drop(list(columnToUse.filter(regex='container_last_seen')),axis=1, inplace=True)
            columnToUse.drop(list(columnToUse.filter(regex='container_start_time_seconds')),axis=1, inplace=True)
            columnToUse.drop(list(columnToUse.filter(regex='container_fs_limit')),axis=1, inplace=True)
            columnToUse.drop(list(columnToUse.filter(regex='container_memory_mapped_filecontainer_memory_mapped_file')),axis=1, inplace=True) 
            columnToUse.drop(list(columnToUse.filter(regex='container_memory_rss')),axis=1, inplace=True)  
            columnToUse.drop(list(columnToUse.filter(regex='container_memory_mapped_file')),axis=1, inplace=True) 
            columnToUse[list(columnToUse.filter(regex='.*_bytes|memory'))] /= 1000000000 #I rescaling everithing is in byte and memory related stats
#            columnToUse = columnToUse.filter(regex='container_memory_cache') 
            ## LSTM data format and rescaling
            
            
#            indexes = []
#            boolA  = columnToUse.min() > 80
#            for o in range(0,len(boolA)):
#                if boolA[o] == True:
#                    print(o)
#                    indexes.append(o)
#            
#            colNames = []
#            for elem in columnToUse.columns:
#                colNames.append(elem.split('{')[0])
#            setColNames = list(set(colNames))
                
            
            input_feature= columnToUse.values
            input_feature_smooth = smooth2(columnToUse,2)
            
#            stdBoolA = input_feature_smooth.std() >= 0.00000001 # we only consider features with std > 0
#            colToSmoothA = input_feature_smooth.columns[stdBoolA == True]
            
            #input_feature_smooth_values = input_feature_smooth.values
            input_feature_smooth_normalized = sc.fit_transform(input_feature_smooth)
            input_feature_df = pd.DataFrame(input_feature_smooth_normalized,columns=input_feature_smooth.columns)
            feature_size = len(input_feature_df.columns)
           
            plt.figure()
            plt.plot(input_feature_smooth_normalized)
            plt.title(train_dataset)
            
            
            ## split into test and validation
            
            SEED = 123 #used to help randomly select the data points
            DATA_SPLIT_PCT = 0.2
            # testo are validation data
            input_data_train, input_data_testo = train_test_split(input_feature_smooth_normalized,test_size=DATA_SPLIT_PCT, random_state=SEED)
            #(input_data_train = input_data
            input_data_train_df = pd.DataFrame(input_data_train,columns=input_feature_df.columns)
            
            
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
            
            
            ##LSTM format testo
            X_testo_A_look=[]
            for i in range(len(input_data_testo)-lookback) :
                t=[]
                for j in range(0,lookback):
                    t.append(input_data_testo[[(i+j)], :])
                X_testo_A_look.append(t)
            
            X_testo= np.array(X_testo_A_look)
          
            X_testo = X_testo.reshape(X_testo.shape[0],lookback, feature_size)
          
         
            callbacks_list = []
            ##TensorBoard to monitor
            tb = TensorBoard(log_dir='.summary_dir',
                            histogram_freq=0,
                            write_graph=True,
                            write_images=True)
            
            
            def create_model(feature_size):
                hidden_layer_size = int(feature_size*0.8)
                print(hidden_layer_size)
                hidden_layer_size2 = int(feature_size*0.5)
                print(hidden_layer_size2)
                lstm_autoencoder = Sequential()
                # Encoder
                lstm_autoencoder.add(LSTM(hidden_layer_size, activation='elu', input_shape=(lookback,feature_size), return_sequences=True, name = 'encode1'))
                lstm_autoencoder.add(Dropout(0.2, name = 'dropout_encode_1'))
                lstm_autoencoder.add(LSTM(hidden_layer_size2, activation='elu', return_sequences=False, name = 'encode2'))
                # lstm_autoencoder.add(LSTM(70, activation='relu', return_sequences=False))
                # lstm_autoencoder.add(Dropout(0.2))
                #lstm_autoencoder.add(LSTM(64, activation='relu', return_sequences=False))
                #lstm_autoencoder.add(LSTM(100, activation='relu', return_sequences=True))
                #lstm_autoencoder.add(LSTM(16, activation='relu', return_sequences=False))
                lstm_autoencoder.add(RepeatVector(lookback))
                # Decoder
                #lstm_autoencoder.add(LSTM(16, activation='relu', return_sequences=True))
                #lstm_autoencoder.add(LSTM(100, activation='relu', return_sequences=True))
                #lstm_autoencoder.add(LSTM(64, activation='relu', return_sequences=True))
                # lstm_autoencoder.add(LSTM(70, activation='relu', return_sequences=True))
                # lstm_autoencoder.add(Dropout(0.2))
                lstm_autoencoder.add(LSTM(hidden_layer_size, activation='elu', return_sequences=True, name = 'dencode1'))
                lstm_autoencoder.add(Dropout(0.2, name = 'dropout_dencode_1'))
                lstm_autoencoder.add(LSTM(feature_size, activation='linear', return_sequences=True, name = 'dencode2'))
                #lstm_autoencoder.add(TimeDistributed(Dense(feature_size)))

                lstm_autoencoder.summary()
                               
            
                lstm_autoencoder.compile(optimizer='adam', loss='mean_squared_error',metrics=['mse'])
                return lstm_autoencoder
            
            
            
            
            
            
            #KerasClassifier
            #lstm_autoencoder = KerasRegressor(build_fn=create_model, verbose=10)
            lstm_autoencoder = create_model(feature_size)
            plot_model(lstm_autoencoder, show_shapes=True,to_file='reconstruct_lstm_autoencoder.png')
            
            ##checkpoint to monitor training
            filepath="weights-improvement-{epoch:02d}-{loss:.2f}.hdf5"
            checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=10, save_best_only=True, mode='min')
            ##Early stopping
            es = EarlyStopping(monitor='loss',patience=5, mode='min')
            
            callbacks_list.append(checkpoint)
            callbacks_list.append(tb)
            callbacks_list.append(es)
            
            
        
            #sinlge train
            lstm_autoencoder_history = lstm_autoencoder.fit(X_train, X_train, validation_split=0.2, epochs=100, batch_size=128,verbose=2,callbacks=callbacks_list).history
        
        
        
#        modelNew = create_model(feature_size)
#        modelNew.load_weights('weights-improvement-83-0.03.hdf5')
        
        def compose_encode(lstm_autoencoder):
            encoder = Sequential()
            encoder.add(lstm_autoencoder.get_layer(name='encode1'))
            encoder.add(lstm_autoencoder.get_layer(name='dropout_encode_1'))
            encoder.add(lstm_autoencoder.get_layer(name='encode2'))
            return encoder
        
        
        encoder = compose_encode(lstm_autoencoder)
        
        encdoed_data = encoder.predict(X_train)
        plt.figure()
        plt.plot(encdoed_data)
             
        
           # import hierarchical clustering libraries
        import scipy.cluster.hierarchy as sch
        from sklearn.cluster import AgglomerativeClustering
     
                ##
        
        # create dendrogram
#        import sys
#        sys.setrecursionlimit(10000)
        dendrogram = sch.dendrogram(sch.linkage(encdoed_data[:2000], method='ward'))
        # create clusters
        hc = AgglomerativeClustering(n_clusters=1, affinity = 'euclidean', linkage = 'ward')
        plt.title('Dendrogram')
        plt.xlabel('Customers')
        plt.ylabel('Euclidean distances')
        plt.show()
        # save clusters for chart
        y_hc = hc.fit_predict(curr_pca)
        plt.figure()
        plt.scatter(curr_pca[y_hc ==0,0], curr_pca[y_hc == 0,1], s=100, c='red')
        plt.scatter(curr_pca[y_hc==1,0], curr_pca[y_hc == 1,1], s=100, c='black')
        plt.scatter(curr_pca[y_hc ==2,0], curr_pca[y_hc == 2,1], s=100, c='blue')
    #    plt.scatter(curr_pca[y_hc ==3,0], curr_pca[y_hc == 3,1], s=100, c='cyan')
        plt.show   
        ###
    
            
        
        
        
        
        
        
        
        # plot train and val loss           
        plt.figure()
        plt.plot(lstm_autoencoder_history['loss'], linewidth=2, label='Train')
        plt.plot(lstm_autoencoder_history['val_loss'], linewidth=2, label='Valid')
        plt.legend(loc='upper right')
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.show()
        
#        plt.figure()
#        plt.plot(lstm_autoencoder_history['mean_squared_error'], linewidth=2, label='Train')
#        plt.plot(lstm_autoencoder_history['val_mean_squared_error'], linewidth=2, label='Valid')
#        plt.show()
        
    
    ## Learn reconstruction error on training to decide anomalt treshold
        
    
        prediction_train = lstm_autoencoder.predict(X_train) #  Train case
        plt.figure()
        plt.plot(flatten(X_train))
        plt.title("Train")
        plt.figure()
        plt.plot(flatten(prediction_train))
        plt.title("Reconstructed")
        
       
        errorDf = pd.DataFrame(flatten(X_train)-flatten(prediction_train),columns = columnToUse.columns)
        errorDf.to_csv("ErrorTraining.csv",index=False)
        
        perFeatureMSE = np.mean(np.power(flatten(X_train)-flatten(prediction_train),2), axis=0)
        plt.figure()
        plt.plot(perFeatureMSE)

#    ##Max error

        toClear = []
        for i in range(0,len(perFeatureMSE)):
            if perFeatureMSE[i] > 0.003:
                toClear.append(i) 
                
        col_to_clear = input_data_train_df.iloc[:,toClear].columns
        aer = df[col_to_clear].std() == 0
        Cnt = 0
        for t in range(0,len(aer)):
            if aer[t] == False:
                Cnt += 1
#        plt.figure()
#        perFeatureMSE.plot(legend=False)
            
        
        scored = pd.DataFrame(index = range(0,X_train.shape[0]))
        scored['Loss_mae'] = np.mean(np.power(flatten(X_train)-flatten(prediction_train),2), axis=1)
        
        #col_mean =  np.mean(np.power(prediction_train-input_feature_df,2), axis=0)
        plt.figure(figsize=(16,9), dpi=80)
        #sns.distplot(scored['Loss_mae'],bins = 2000, kde = True, color = 'blue',norm_hist=True)
        thresholds = [ scored['Loss_mae'].mean()+scored['Loss_mae'].std(),  scored['Loss_mae'].mean()+2*scored['Loss_mae'].std(),  scored['Loss_mae'].mean()+3*scored['Loss_mae'].std()]
        scored['Threshold1'] = thresholds[0]
        scored['Anomaly1'] = scored['Loss_mae'] > scored ['Threshold1']
        scored['Threshold2'] =  thresholds[1]
        scored['Anomaly2'] = scored['Loss_mae'] > scored ['Threshold2']
        scored['Threshold3'] =  thresholds[2]
        scored['Anomaly3'] = scored['Loss_mae'] > scored ['Threshold3']
        #scored.head()
        scored.plot(title='Train')
    
            
        predicted_values = []
        input_data_array = []
        test_dataset = 'baseline6'
        files2 = [open(type+test_dataset+"/"+file, 'r') for file in os.listdir(type+test_dataset)]
        for file_handler2 in files2: #iterates on toAnalyze files
            df2 = pd.DataFrame()
            df2 = pd.read_csv(file_handler2, sep = ";", header = 0,low_memory=False, index_col=None, error_bad_lines=False)
            df2 = df2.select_dtypes(exclude=['object'])
            df2.drop(list(df2.filter(regex='Unnamed')),axis=1, inplace=True)
            df2.drop(list(df2.filter(regex='timestamp')),axis=1, inplace=True)
            #df2.drop(list(df.filter(regex='memory')),axis=1, inplace=True)
                 
            
            
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
                     
            df2=df2.fillna(value=0) # first elem will be Nan after diff
            reverse_df2 =  df2.iloc[::-1]  # reverse df for lookback
        
            # Gauges pre-processing
            ## nothing to do
                
            # Normalization
            columnToUse2 = reverse_df2.filter(columnToUse.columns) #use same columns as in training phase
            columnToUse2[list(columnToUse2.filter(regex='.*_bytes|memory'))] /= 1000000000 #I rescaling everithing is in byte and memory related stats
            input_feature2_smooth = smooth2(columnToUse2,2)
#            input_feature2_smooth.reindex(axis='index')
            #input_feature2_smooth_values = input_feature2_smooth.values
#            input_feature2_df = pd.DataFrame(input_feature2,columns = columnToUse2.columns)
#            stdBool2 = df2.std() <= 0.001
#            colToSmooth2 = df2.columns[stdBool2 == True]
#            colToSmoothIndex2 = [df2.columns.get_loc(c) for c in colToSmooth2]
         
#            input_feature2_smooth_df = pd.DataFrame(input_feature2_smooth,columns = columnToUse2.columns)
            
#            for i in range(0,input_feature2_smooth.shape[1]):
#                if sc.var_[i] != 0:
#                    input_feature2_smooth.iloc[:,i] = (input_feature2_smooth.iloc[:,i]-sc.mean_[i])/sc.var_[i]
#                else:
#                     input_feature2_smooth.iloc[:,i] = (input_feature2_smooth.iloc[:,i]-sc.mean_[i])
            
            #dfConcat = pd.concat((df,df2),ignore_index=True,sort=False)
            input_feature2_smooth_normalized = sc.transform(input_feature2_smooth)

#            plt.figure()
#            plt.plot(input_feature2_smooth_normalized)
#            plt.title(test_dataset)
            
          
            
            input_feature2_smooth_normalized_df = pd.DataFrame(input_feature2_smooth_normalized, columns= columnToUse2.columns)
            indexes = []
            boolA  = input_feature2_smooth_normalized_df.max() < -1000
            for o in range(0,len(boolA)):
                if boolA[o] == True:
                    print(o)
                    indexes.append(o)
#            bolT = input_feature2_smooth_normalized_df.mean() > 15
#            la = []
#            for i in range(0,len(bolT)):
#                if bolT[i] == True:
#                    print(i)
#                    la.append(i)
#            input_feature2_smooth_normalized_df.iloc[:,la].columns
            
            
#            input_feature2_smooth_normalized = input_feature2_smooth.values

            ## LSTM data format preparation
        
            X2=[]
            ##LSTM format training
            X_test_A_look=[]
            for i in range(len(input_feature2_smooth_normalized)-lookback) :
                t=[]
                for j in range(0,lookback):
                    t.append(input_feature2_smooth_normalized[[(i+j)], :])
                X_test_A_look.append(t)
            
            X= np.array(X_test_A_look)
          
            X_test = X.reshape(X.shape[0],lookback, feature_size)
             
           
            #prediction_test = modelNew.predict(X_test) #  Test case
            prediction_test = lstm_autoencoder.predict(X_test) #  Test case
            plt.figure()
            plt.plot(flatten(prediction_test))
            plt.title("reconstructed")
            plt.figure()
            plt.plot(flatten(X_test))  
            plt.title("Test")
            
            
            perFeatureMSE2 = np.mean(np.power(flatten(X_test)-flatten(prediction_test),2), axis=0)
            plt.figure()
            plt.plot(perFeatureMSE2)
            
#            pd.DataFrame(perFeatureMSE2).to_csv("TestBaseline_stress_baseline_mean_squared_linear_2Layer180_80.csv")
            
            
            
            
#            errorArray2 = flatten(X_test)- flatten(prediction_test)
#            errorDf = pd.DataFrame(errorArray2, columns = columnToUse.columns)
#            
#            x_surface = []
#            y_surface = []
#            z_surface = []
#            for x in range(1,feature_size+1):
#                for y in range(1,int((errorArray2.shape[0]+1)/4)):
#                    x_surface.append(x)
#                    y_surface.append(y)
#                    z_surface.append(int(errorArray2[y-1][x-1]))
#            
#            fig = plt.figure()
#            ax = fig.gca(projection='3d')
#            
#            ax.plot_trisurf(x_surface, y_surface, z_surface, linewidth=0.2, antialiased=True)
#            

            scored2 = pd.DataFrame(index = range(0,X_test.shape[0]))
#            prediction_test = flatten(prediction_test)
#            prediction_test = pd.DataFrame(prediction_test, columns=input_feature_df2.columns)
#            scored = pd.DataFrame(index = input_feature_df.index)
#            Xtest = input_feature_df2 #[0:len(input_data2)-lookback-1,]
            scored2['Loss_mae'] = np.mean(np.power(flatten(X_test)-flatten(prediction_test),2), axis=1)
            
            plt.figure(figsize=(16,9), dpi=80)
            #sns.distplot(scored2['Loss_mae'],bins = 20000, kde = True, color = 'blue',norm_hist=True)
            #thresholds = [ scored['Loss_mae'].mean()+scored['Loss_mae'].std(),  scored['Loss_mae'].mean()+2*scored['Loss_mae'].std(),  scored['Loss_mae'].mean()+3*scored['Loss_mae'].std()]
            scored2['Threshold1'] = thresholds[0]
            scored2['Anomaly1'] = scored['Loss_mae'] > scored ['Threshold1']
            scored2['Threshold2'] =  thresholds[1]
            scored2['Anomaly2'] = scored['Loss_mae'] > scored ['Threshold2']
            scored2['Threshold3'] =  thresholds[2]
            scored2['Anomaly3'] = scored['Loss_mae'] > scored ['Threshold3']
            #scored.head()
            scored2.plot(title='Test')
           
            
            errorDfTest = pd.DataFrame(flatten(X_test)-flatten(prediction_test),columns = columnToUse.columns)
            errorDfTest.to_csv("ErrorTest.csv",index=False)
            
            maxIndexTest =  errorDfTest.idxmax()
            maxCall  = maxIndexTest < 23000
            for o in range(0,len(maxCall)):
                if maxCall[o] == True:
                    print(o)
            
          
            
            
            
            
#            scored['Threshold1'] = 3
#            #scored['Anomaly1'] = scored['Loss_mae'] > scored ['Threshold1']
#        #    scored['Threshold2'] =  thresholds[1]
#        #    scored['Anomaly2'] = scored['Loss_mae'] > scored ['Threshold2']
#        #    scored['Threshold3'] =  thresholds[2]
#        #    scored['Anomaly3'] = scored['Loss_mae'] > scored ['Threshold3']
#        #    scored.head()
#            plt.figure()
#            scored.plot(title = "Test "+file_handler2.name.split("/")[1].split(".")[0] )
#            plt.axvline(x=347, color='r')
#            plt.axvline(x=687, color='r')
#            plt.axvline(x=1074, color='r')
#            plt.axvline(x=1437, color='r')
#            plt.axvline(x=1854, color='r')
#            
#            perFeatureMSE2 = np.mean(np.power(input_feature_df2-prediction_test,2), axis=0)
#            plt.figure()
#            perFeatureMSE2.plot(legend=False)
#            perFeatureMSE2.to_csv("NewTest_mean_squared_logarithmic_error_1Layer80.csv")
#            
#            
#            plt.figure()
#            input_feature_df.plot(legend=False)
#            plt.figure()
#            input_feature_df2.plot(legend=False)
#            
#            
#            
#            ##Max error
#            testMaxError = np.power(prediction_test-Xtest,2).max()
#            plt.figure()
#            testMaxError.plot(legend=False)
#            #plt.gca().invert_xaxis() 
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
                  ##############
            
#            testDrop = ['container_cpu_system_seconds_total{container="",container_name="",id="/kubepods/besteffort/pod1d8d6010-c663-49b3-9000-6483e8d6724c",image="",name="",namespace="openims",pod="pcscf-deployment-84649d986b-mk8jp",pod_name="pcscf-deployment-84649d986b-mk8jp"}',
#       'container_cpu_system_seconds_total{container="",container_name="",id="/kubepods/besteffort/pod3067e0c2-81d0-47a6-bdc3-2a458ca24b3b",image="",name="",namespace="openims",pod="dns-deployment-6d486c4fb6-slsq6",pod_name="dns-deployment-6d486c4fb6-slsq6"}',
#       'container_cpu_system_seconds_total{container="",container_name="",id="/kubepods/besteffort/pod5e0c22bb-f250-4c4f-b2f5-d90e91266229",image="",name="",namespace="openims",pod="scscf-deployment-56679d845-cnjr7",pod_name="scscf-deployment-56679d845-cnjr7"}',
#       'container_cpu_system_seconds_total{container="",container_name="",id="/kubepods/besteffort/podae951ca5-1dca-4268-af5a-8efc397d5cda",image="",name="",namespace="openims",pod="icscf-deployment-74f5bb8559-wppfp",pod_name="icscf-deployment-74f5bb8559-wppfp"}',
#       'container_cpu_system_seconds_total{container="",container_name="",id="/kubepods/besteffort/podd2703fdd-859e-4812-bdcd-282be0659c3e",image="",name="",namespace="openims",pod="hss-deployment-7b9c7786fd-qgvts",pod_name="hss-deployment-7b9c7786fd-qgvts"}',
#       'container_cpu_system_seconds_total{container="dns",container_name="dns",id="/kubepods/besteffort/pod3067e0c2-81d0-47a6-bdc3-2a458ca24b3b/83a095b635a5cd18d05cf2ef7ca854f046c85a63105a1e4fa7917e7dddb7de2b",image="alessiodiama/pcscf@sha256:89c75c66a732d6324b0f11c57b8dfd78ab5efd6335d9a3bc5959e6c191e3f564",name="k8s_dns_dns-deployment-6d486c4fb6-slsq6_openims_3067e0c2-81d0-47a6-bdc3-2a458ca24b3b_7",namespace="openims",pod="dns-deployment-6d486c4fb6-slsq6",pod_name="dns-deployment-6d486c4fb6-slsq6"}',
#       'container_cpu_system_seconds_total{container="icscf",container_name="icscf",id="/kubepods/besteffort/podae951ca5-1dca-4268-af5a-8efc397d5cda/a73e943ea5bf113f1983f0211457274f737f9bb26bad85501ed244836d889d20",image="alessiodiama/icscf@sha256:15c557e7be333c37e46e17ff2c5be287d0658c2d3c280e45d4390f0217d1863c",name="k8s_icscf_icscf-deployment-74f5bb8559-wppfp_openims_ae951ca5-1dca-4268-af5a-8efc397d5cda_15",namespace="openims",pod="icscf-deployment-74f5bb8559-wppfp",pod_name="icscf-deployment-74f5bb8559-wppfp"}',
#       'container_cpu_system_seconds_total{container="pcscf",container_name="pcscf",id="/kubepods/besteffort/pod1d8d6010-c663-49b3-9000-6483e8d6724c/dc95a57bc0260c4aeeb287c99c95828a5a5ab31505c90588536aac9a71717c11",image="alessiodiama/pcscf@sha256:89c75c66a732d6324b0f11c57b8dfd78ab5efd6335d9a3bc5959e6c191e3f564",name="k8s_pcscf_pcscf-deployment-84649d986b-mk8jp_openims_1d8d6010-c663-49b3-9000-6483e8d6724c_12",namespace="openims",pod="pcscf-deployment-84649d986b-mk8jp",pod_name="pcscf-deployment-84649d986b-mk8jp"}',
#       'container_cpu_system_seconds_total{container="pcscf",container_name="pcscf",id="/kubepods/besteffort/podd2703fdd-859e-4812-bdcd-282be0659c3e/733235a0676cbdd952c54e580bfc80e9e876b7b52e308444a19849f6bf1d8af8",image="alessiodiama/icscf@sha256:15c557e7be333c37e46e17ff2c5be287d0658c2d3c280e45d4390f0217d1863c",name="k8s_pcscf_hss-deployment-7b9c7786fd-qgvts_openims_d2703fdd-859e-4812-bdcd-282be0659c3e_15",namespace="openims",pod="hss-deployment-7b9c7786fd-qgvts",pod_name="hss-deployment-7b9c7786fd-qgvts"}',
#       'container_cpu_system_seconds_total{container="scscf",container_name="scscf",id="/kubepods/besteffort/pod5e0c22bb-f250-4c4f-b2f5-d90e91266229/75dd0794a46c4fc3c62feba93765022855982e5793119eb6b0db8375e37fff13",image="alessiodiama/scscf@sha256:a4b9c179ca81b9d5fda7e0511d2bf334cc97167d7508365926f1247ac4941f45",name="k8s_scscf_scscf-deployment-56679d845-cnjr7_openims_5e0c22bb-f250-4c4f-b2f5-d90e91266229_15",namespace="openims",pod="scscf-deployment-56679d845-cnjr7",pod_name="scscf-deployment-56679d845-cnjr7"}',
#       'container_cpu_usage_seconds_total{container="",container_name="",cpu="total",id="/kubepods/besteffort/pod1d8d6010-c663-49b3-9000-6483e8d6724c",image="",name="",namespace="openims",pod="pcscf-deployment-84649d986b-mk8jp",pod_name="pcscf-deployment-84649d986b-mk8jp"}',
#       'container_cpu_usage_seconds_total{container="",container_name="",cpu="total",id="/kubepods/besteffort/pod3067e0c2-81d0-47a6-bdc3-2a458ca24b3b",image="",name="",namespace="openims",pod="dns-deployment-6d486c4fb6-slsq6",pod_name="dns-deployment-6d486c4fb6-slsq6"}',
#       'container_cpu_usage_seconds_total{container="",container_name="",cpu="total",id="/kubepods/besteffort/pod5e0c22bb-f250-4c4f-b2f5-d90e91266229",image="",name="",namespace="openims",pod="scscf-deployment-56679d845-cnjr7",pod_name="scscf-deployment-56679d845-cnjr7"}',
#       'container_cpu_usage_seconds_total{container="",container_name="",cpu="total",id="/kubepods/besteffort/podae951ca5-1dca-4268-af5a-8efc397d5cda",image="",name="",namespace="openims",pod="icscf-deployment-74f5bb8559-wppfp",pod_name="icscf-deployment-74f5bb8559-wppfp"}',
#       'container_cpu_usage_seconds_total{container="",container_name="",cpu="total",id="/kubepods/besteffort/podd2703fdd-859e-4812-bdcd-282be0659c3e",image="",name="",namespace="openims",pod="hss-deployment-7b9c7786fd-qgvts",pod_name="hss-deployment-7b9c7786fd-qgvts"}',
#       'container_cpu_usage_seconds_total{container="dns",container_name="dns",cpu="total",id="/kubepods/besteffort/pod3067e0c2-81d0-47a6-bdc3-2a458ca24b3b/83a095b635a5cd18d05cf2ef7ca854f046c85a63105a1e4fa7917e7dddb7de2b",image="alessiodiama/pcscf@sha256:89c75c66a732d6324b0f11c57b8dfd78ab5efd6335d9a3bc5959e6c191e3f564",name="k8s_dns_dns-deployment-6d486c4fb6-slsq6_openims_3067e0c2-81d0-47a6-bdc3-2a458ca24b3b_7",namespace="openims",pod="dns-deployment-6d486c4fb6-slsq6",pod_name="dns-deployment-6d486c4fb6-slsq6"}',
#       'container_cpu_usage_seconds_total{container="icscf",container_name="icscf",cpu="total",id="/kubepods/besteffort/podae951ca5-1dca-4268-af5a-8efc397d5cda/a73e943ea5bf113f1983f0211457274f737f9bb26bad85501ed244836d889d20",image="alessiodiama/icscf@sha256:15c557e7be333c37e46e17ff2c5be287d0658c2d3c280e45d4390f0217d1863c",name="k8s_icscf_icscf-deployment-74f5bb8559-wppfp_openims_ae951ca5-1dca-4268-af5a-8efc397d5cda_15",namespace="openims",pod="icscf-deployment-74f5bb8559-wppfp",pod_name="icscf-deployment-74f5bb8559-wppfp"}',
#       'container_cpu_usage_seconds_total{container="pcscf",container_name="pcscf",cpu="total",id="/kubepods/besteffort/pod1d8d6010-c663-49b3-9000-6483e8d6724c/dc95a57bc0260c4aeeb287c99c95828a5a5ab31505c90588536aac9a71717c11",image="alessiodiama/pcscf@sha256:89c75c66a732d6324b0f11c57b8dfd78ab5efd6335d9a3bc5959e6c191e3f564",name="k8s_pcscf_pcscf-deployment-84649d986b-mk8jp_openims_1d8d6010-c663-49b3-9000-6483e8d6724c_12",namespace="openims",pod="pcscf-deployment-84649d986b-mk8jp",pod_name="pcscf-deployment-84649d986b-mk8jp"}',
#       'container_cpu_usage_seconds_total{container="pcscf",container_name="pcscf",cpu="total",id="/kubepods/besteffort/podd2703fdd-859e-4812-bdcd-282be0659c3e/733235a0676cbdd952c54e580bfc80e9e876b7b52e308444a19849f6bf1d8af8",image="alessiodiama/icscf@sha256:15c557e7be333c37e46e17ff2c5be287d0658c2d3c280e45d4390f0217d1863c",name="k8s_pcscf_hss-deployment-7b9c7786fd-qgvts_openims_d2703fdd-859e-4812-bdcd-282be0659c3e_15",namespace="openims",pod="hss-deployment-7b9c7786fd-qgvts",pod_name="hss-deployment-7b9c7786fd-qgvts"}',
#       'container_cpu_usage_seconds_total{container="scscf",container_name="scscf",cpu="total",id="/kubepods/besteffort/pod5e0c22bb-f250-4c4f-b2f5-d90e91266229/75dd0794a46c4fc3c62feba93765022855982e5793119eb6b0db8375e37fff13",image="alessiodiama/scscf@sha256:a4b9c179ca81b9d5fda7e0511d2bf334cc97167d7508365926f1247ac4941f45",name="k8s_scscf_scscf-deployment-56679d845-cnjr7_openims_5e0c22bb-f250-4c4f-b2f5-d90e91266229_15",namespace="openims",pod="scscf-deployment-56679d845-cnjr7",pod_name="scscf-deployment-56679d845-cnjr7"}',
#       'container_cpu_user_seconds_total{container="",container_name="",id="/kubepods/besteffort/pod1d8d6010-c663-49b3-9000-6483e8d6724c",image="",name="",namespace="openims",pod="pcscf-deployment-84649d986b-mk8jp",pod_name="pcscf-deployment-84649d986b-mk8jp"}',
#       'container_cpu_user_seconds_total{container="",container_name="",id="/kubepods/besteffort/pod3067e0c2-81d0-47a6-bdc3-2a458ca24b3b",image="",name="",namespace="openims",pod="dns-deployment-6d486c4fb6-slsq6",pod_name="dns-deployment-6d486c4fb6-slsq6"}',
#       'container_cpu_user_seconds_total{container="",container_name="",id="/kubepods/besteffort/pod5e0c22bb-f250-4c4f-b2f5-d90e91266229",image="",name="",namespace="openims",pod="scscf-deployment-56679d845-cnjr7",pod_name="scscf-deployment-56679d845-cnjr7"}',
#       'container_cpu_user_seconds_total{container="",container_name="",id="/kubepods/besteffort/podae951ca5-1dca-4268-af5a-8efc397d5cda",image="",name="",namespace="openims",pod="icscf-deployment-74f5bb8559-wppfp",pod_name="icscf-deployment-74f5bb8559-wppfp"}',
#       'container_cpu_user_seconds_total{container="",container_name="",id="/kubepods/besteffort/podd2703fdd-859e-4812-bdcd-282be0659c3e",image="",name="",namespace="openims",pod="hss-deployment-7b9c7786fd-qgvts",pod_name="hss-deployment-7b9c7786fd-qgvts"}',
#       'container_cpu_user_seconds_total{container="dns",container_name="dns",id="/kubepods/besteffort/pod3067e0c2-81d0-47a6-bdc3-2a458ca24b3b/83a095b635a5cd18d05cf2ef7ca854f046c85a63105a1e4fa7917e7dddb7de2b",image="alessiodiama/pcscf@sha256:89c75c66a732d6324b0f11c57b8dfd78ab5efd6335d9a3bc5959e6c191e3f564",name="k8s_dns_dns-deployment-6d486c4fb6-slsq6_openims_3067e0c2-81d0-47a6-bdc3-2a458ca24b3b_7",namespace="openims",pod="dns-deployment-6d486c4fb6-slsq6",pod_name="dns-deployment-6d486c4fb6-slsq6"}',
#       'container_cpu_user_seconds_total{container="icscf",container_name="icscf",id="/kubepods/besteffort/podae951ca5-1dca-4268-af5a-8efc397d5cda/a73e943ea5bf113f1983f0211457274f737f9bb26bad85501ed244836d889d20",image="alessiodiama/icscf@sha256:15c557e7be333c37e46e17ff2c5be287d0658c2d3c280e45d4390f0217d1863c",name="k8s_icscf_icscf-deployment-74f5bb8559-wppfp_openims_ae951ca5-1dca-4268-af5a-8efc397d5cda_15",namespace="openims",pod="icscf-deployment-74f5bb8559-wppfp",pod_name="icscf-deployment-74f5bb8559-wppfp"}',
#       'container_cpu_user_seconds_total{container="pcscf",container_name="pcscf",id="/kubepods/besteffort/pod1d8d6010-c663-49b3-9000-6483e8d6724c/dc95a57bc0260c4aeeb287c99c95828a5a5ab31505c90588536aac9a71717c11",image="alessiodiama/pcscf@sha256:89c75c66a732d6324b0f11c57b8dfd78ab5efd6335d9a3bc5959e6c191e3f564",name="k8s_pcscf_pcscf-deployment-84649d986b-mk8jp_openims_1d8d6010-c663-49b3-9000-6483e8d6724c_12",namespace="openims",pod="pcscf-deployment-84649d986b-mk8jp",pod_name="pcscf-deployment-84649d986b-mk8jp"}',
#       'container_cpu_user_seconds_total{container="pcscf",container_name="pcscf",id="/kubepods/besteffort/podd2703fdd-859e-4812-bdcd-282be0659c3e/733235a0676cbdd952c54e580bfc80e9e876b7b52e308444a19849f6bf1d8af8",image="alessiodiama/icscf@sha256:15c557e7be333c37e46e17ff2c5be287d0658c2d3c280e45d4390f0217d1863c",name="k8s_pcscf_hss-deployment-7b9c7786fd-qgvts_openims_d2703fdd-859e-4812-bdcd-282be0659c3e_15",namespace="openims",pod="hss-deployment-7b9c7786fd-qgvts",pod_name="hss-deployment-7b9c7786fd-qgvts"}',
#       'container_cpu_user_seconds_total{container="scscf",container_name="scscf",id="/kubepods/besteffort/pod5e0c22bb-f250-4c4f-b2f5-d90e91266229/75dd0794a46c4fc3c62feba93765022855982e5793119eb6b0db8375e37fff13",image="alessiodiama/scscf@sha256:a4b9c179ca81b9d5fda7e0511d2bf334cc97167d7508365926f1247ac4941f45",name="k8s_scscf_scscf-deployment-56679d845-cnjr7_openims_5e0c22bb-f250-4c4f-b2f5-d90e91266229_15",namespace="openims",pod="scscf-deployment-56679d845-cnjr7",pod_name="scscf-deployment-56679d845-cnjr7"}',
#       'container_fs_reads_bytes_total{container="pcscf",container_name="pcscf",device="/dev/dm-0",id="/kubepods/besteffort/podd2703fdd-859e-4812-bdcd-282be0659c3e/733235a0676cbdd952c54e580bfc80e9e876b7b52e308444a19849f6bf1d8af8",image="alessiodiama/icscf@sha256:15c557e7be333c37e46e17ff2c5be287d0658c2d3c280e45d4390f0217d1863c",name="k8s_pcscf_hss-deployment-7b9c7786fd-qgvts_openims_d2703fdd-859e-4812-bdcd-282be0659c3e_15",namespace="openims",pod="hss-deployment-7b9c7786fd-qgvts",pod_name="hss-deployment-7b9c7786fd-qgvts"}',
#       'container_fs_reads_bytes_total{container="pcscf",container_name="pcscf",device="/dev/sda",id="/kubepods/besteffort/podd2703fdd-859e-4812-bdcd-282be0659c3e/733235a0676cbdd952c54e580bfc80e9e876b7b52e308444a19849f6bf1d8af8",image="alessiodiama/icscf@sha256:15c557e7be333c37e46e17ff2c5be287d0658c2d3c280e45d4390f0217d1863c",name="k8s_pcscf_hss-deployment-7b9c7786fd-qgvts_openims_d2703fdd-859e-4812-bdcd-282be0659c3e_15",namespace="openims",pod="hss-deployment-7b9c7786fd-qgvts",pod_name="hss-deployment-7b9c7786fd-qgvts"}',
#       'container_fs_reads_total{container="pcscf",container_name="pcscf",device="/dev/dm-0",id="/kubepods/besteffort/podd2703fdd-859e-4812-bdcd-282be0659c3e/733235a0676cbdd952c54e580bfc80e9e876b7b52e308444a19849f6bf1d8af8",image="alessiodiama/icscf@sha256:15c557e7be333c37e46e17ff2c5be287d0658c2d3c280e45d4390f0217d1863c",name="k8s_pcscf_hss-deployment-7b9c7786fd-qgvts_openims_d2703fdd-859e-4812-bdcd-282be0659c3e_15",namespace="openims",pod="hss-deployment-7b9c7786fd-qgvts",pod_name="hss-deployment-7b9c7786fd-qgvts"}',
#       'container_fs_reads_total{container="pcscf",container_name="pcscf",device="/dev/sda",id="/kubepods/besteffort/podd2703fdd-859e-4812-bdcd-282be0659c3e/733235a0676cbdd952c54e580bfc80e9e876b7b52e308444a19849f6bf1d8af8",image="alessiodiama/icscf@sha256:15c557e7be333c37e46e17ff2c5be287d0658c2d3c280e45d4390f0217d1863c",name="k8s_pcscf_hss-deployment-7b9c7786fd-qgvts_openims_d2703fdd-859e-4812-bdcd-282be0659c3e_15",namespace="openims",pod="hss-deployment-7b9c7786fd-qgvts",pod_name="hss-deployment-7b9c7786fd-qgvts"}',
#       'container_fs_writes_bytes_total{container="dns",container_name="dns",device="/dev/dm-0",id="/kubepods/besteffort/pod3067e0c2-81d0-47a6-bdc3-2a458ca24b3b/83a095b635a5cd18d05cf2ef7ca854f046c85a63105a1e4fa7917e7dddb7de2b",image="alessiodiama/pcscf@sha256:89c75c66a732d6324b0f11c57b8dfd78ab5efd6335d9a3bc5959e6c191e3f564",name="k8s_dns_dns-deployment-6d486c4fb6-slsq6_openims_3067e0c2-81d0-47a6-bdc3-2a458ca24b3b_7",namespace="openims",pod="dns-deployment-6d486c4fb6-slsq6",pod_name="dns-deployment-6d486c4fb6-slsq6"}',
#       'container_fs_writes_bytes_total{container="dns",container_name="dns",device="/dev/sda",id="/kubepods/besteffort/pod3067e0c2-81d0-47a6-bdc3-2a458ca24b3b/83a095b635a5cd18d05cf2ef7ca854f046c85a63105a1e4fa7917e7dddb7de2b",image="alessiodiama/pcscf@sha256:89c75c66a732d6324b0f11c57b8dfd78ab5efd6335d9a3bc5959e6c191e3f564",name="k8s_dns_dns-deployment-6d486c4fb6-slsq6_openims_3067e0c2-81d0-47a6-bdc3-2a458ca24b3b_7",namespace="openims",pod="dns-deployment-6d486c4fb6-slsq6",pod_name="dns-deployment-6d486c4fb6-slsq6"}',
#       'container_fs_writes_total{container="dns",container_name="dns",device="/dev/dm-0",id="/kubepods/besteffort/pod3067e0c2-81d0-47a6-bdc3-2a458ca24b3b/83a095b635a5cd18d05cf2ef7ca854f046c85a63105a1e4fa7917e7dddb7de2b",image="alessiodiama/pcscf@sha256:89c75c66a732d6324b0f11c57b8dfd78ab5efd6335d9a3bc5959e6c191e3f564",name="k8s_dns_dns-deployment-6d486c4fb6-slsq6_openims_3067e0c2-81d0-47a6-bdc3-2a458ca24b3b_7",namespace="openims",pod="dns-deployment-6d486c4fb6-slsq6",pod_name="dns-deployment-6d486c4fb6-slsq6"}',
#       'container_fs_writes_total{container="dns",container_name="dns",device="/dev/sda",id="/kubepods/besteffort/pod3067e0c2-81d0-47a6-bdc3-2a458ca24b3b/83a095b635a5cd18d05cf2ef7ca854f046c85a63105a1e4fa7917e7dddb7de2b",image="alessiodiama/pcscf@sha256:89c75c66a732d6324b0f11c57b8dfd78ab5efd6335d9a3bc5959e6c191e3f564",name="k8s_dns_dns-deployment-6d486c4fb6-slsq6_openims_3067e0c2-81d0-47a6-bdc3-2a458ca24b3b_7",namespace="openims",pod="dns-deployment-6d486c4fb6-slsq6",pod_name="dns-deployment-6d486c4fb6-slsq6"}',
#       'container_network_receive_bytes_total{container="POD",container_name="POD",id="/kubepods/besteffort/pod1d8d6010-c663-49b3-9000-6483e8d6724c/ed7e2a90b6417b69b10c607d2d2098f042baed5acd1aaccd48761a67fc61157c",image="k8s.gcr.io/pause:3.1",interface="eth0",name="k8s_POD_pcscf-deployment-84649d986b-mk8jp_openims_1d8d6010-c663-49b3-9000-6483e8d6724c_5",namespace="openims",pod="pcscf-deployment-84649d986b-mk8jp",pod_name="pcscf-deployment-84649d986b-mk8jp"}',
#       'container_network_receive_bytes_total{container="POD",container_name="POD",id="/kubepods/besteffort/pod3067e0c2-81d0-47a6-bdc3-2a458ca24b3b/77e9f81f120a4cd1106add9fe98523d1537ce2f576490fb58ba711496f6e0b38",image="k8s.gcr.io/pause:3.1",interface="eth0",name="k8s_POD_dns-deployment-6d486c4fb6-slsq6_openims_3067e0c2-81d0-47a6-bdc3-2a458ca24b3b_4",namespace="openims",pod="dns-deployment-6d486c4fb6-slsq6",pod_name="dns-deployment-6d486c4fb6-slsq6"}',
#       'container_network_receive_bytes_total{container="POD",container_name="POD",id="/kubepods/besteffort/pod5e0c22bb-f250-4c4f-b2f5-d90e91266229/50f111343380e542220175129525047a784b17efef855af70b0be18ea3588c9e",image="k8s.gcr.io/pause:3.1",interface="eth0",name="k8s_POD_scscf-deployment-56679d845-cnjr7_openims_5e0c22bb-f250-4c4f-b2f5-d90e91266229_4",namespace="openims",pod="scscf-deployment-56679d845-cnjr7",pod_name="scscf-deployment-56679d845-cnjr7"}',
#       'container_network_receive_bytes_total{container="POD",container_name="POD",id="/kubepods/besteffort/podae951ca5-1dca-4268-af5a-8efc397d5cda/33867386d82e99c8e83c6ff82038de4f25d676c1f351e3457ae819235078e27e",image="k8s.gcr.io/pause:3.1",interface="eth0",name="k8s_POD_icscf-deployment-74f5bb8559-wppfp_openims_ae951ca5-1dca-4268-af5a-8efc397d5cda_5",namespace="openims",pod="icscf-deployment-74f5bb8559-wppfp",pod_name="icscf-deployment-74f5bb8559-wppfp"}',
#       'container_network_receive_bytes_total{container="POD",container_name="POD",id="/kubepods/besteffort/podd2703fdd-859e-4812-bdcd-282be0659c3e/d8ee6209d0b888c34cdb89bab93974834326915842882831304eba5564db14ec",image="k8s.gcr.io/pause:3.1",interface="eth0",name="k8s_POD_hss-deployment-7b9c7786fd-qgvts_openims_d2703fdd-859e-4812-bdcd-282be0659c3e_17",namespace="openims",pod="hss-deployment-7b9c7786fd-qgvts",pod_name="hss-deployment-7b9c7786fd-qgvts"}',
#       'container_network_receive_packets_total{container="POD",container_name="POD",id="/kubepods/besteffort/pod1d8d6010-c663-49b3-9000-6483e8d6724c/ed7e2a90b6417b69b10c607d2d2098f042baed5acd1aaccd48761a67fc61157c",image="k8s.gcr.io/pause:3.1",interface="eth0",name="k8s_POD_pcscf-deployment-84649d986b-mk8jp_openims_1d8d6010-c663-49b3-9000-6483e8d6724c_5",namespace="openims",pod="pcscf-deployment-84649d986b-mk8jp",pod_name="pcscf-deployment-84649d986b-mk8jp"}',
#       'container_network_receive_packets_total{container="POD",container_name="POD",id="/kubepods/besteffort/pod3067e0c2-81d0-47a6-bdc3-2a458ca24b3b/77e9f81f120a4cd1106add9fe98523d1537ce2f576490fb58ba711496f6e0b38",image="k8s.gcr.io/pause:3.1",interface="eth0",name="k8s_POD_dns-deployment-6d486c4fb6-slsq6_openims_3067e0c2-81d0-47a6-bdc3-2a458ca24b3b_4",namespace="openims",pod="dns-deployment-6d486c4fb6-slsq6",pod_name="dns-deployment-6d486c4fb6-slsq6"}',
#       'container_network_receive_packets_total{container="POD",container_name="POD",id="/kubepods/besteffort/pod5e0c22bb-f250-4c4f-b2f5-d90e91266229/50f111343380e542220175129525047a784b17efef855af70b0be18ea3588c9e",image="k8s.gcr.io/pause:3.1",interface="eth0",name="k8s_POD_scscf-deployment-56679d845-cnjr7_openims_5e0c22bb-f250-4c4f-b2f5-d90e91266229_4",namespace="openims",pod="scscf-deployment-56679d845-cnjr7",pod_name="scscf-deployment-56679d845-cnjr7"}',
#       'container_network_receive_packets_total{container="POD",container_name="POD",id="/kubepods/besteffort/podae951ca5-1dca-4268-af5a-8efc397d5cda/33867386d82e99c8e83c6ff82038de4f25d676c1f351e3457ae819235078e27e",image="k8s.gcr.io/pause:3.1",interface="eth0",name="k8s_POD_icscf-deployment-74f5bb8559-wppfp_openims_ae951ca5-1dca-4268-af5a-8efc397d5cda_5",namespace="openims",pod="icscf-deployment-74f5bb8559-wppfp",pod_name="icscf-deployment-74f5bb8559-wppfp"}',
#       'container_network_receive_packets_total{container="POD",container_name="POD",id="/kubepods/besteffort/podd2703fdd-859e-4812-bdcd-282be0659c3e/d8ee6209d0b888c34cdb89bab93974834326915842882831304eba5564db14ec",image="k8s.gcr.io/pause:3.1",interface="eth0",name="k8s_POD_hss-deployment-7b9c7786fd-qgvts_openims_d2703fdd-859e-4812-bdcd-282be0659c3e_17",namespace="openims",pod="hss-deployment-7b9c7786fd-qgvts",pod_name="hss-deployment-7b9c7786fd-qgvts"}',
#       'container_network_transmit_bytes_total{container="POD",container_name="POD",id="/kubepods/besteffort/pod1d8d6010-c663-49b3-9000-6483e8d6724c/ed7e2a90b6417b69b10c607d2d2098f042baed5acd1aaccd48761a67fc61157c",image="k8s.gcr.io/pause:3.1",interface="eth0",name="k8s_POD_pcscf-deployment-84649d986b-mk8jp_openims_1d8d6010-c663-49b3-9000-6483e8d6724c_5",namespace="openims",pod="pcscf-deployment-84649d986b-mk8jp",pod_name="pcscf-deployment-84649d986b-mk8jp"}',
#       'container_network_transmit_bytes_total{container="POD",container_name="POD",id="/kubepods/besteffort/pod3067e0c2-81d0-47a6-bdc3-2a458ca24b3b/77e9f81f120a4cd1106add9fe98523d1537ce2f576490fb58ba711496f6e0b38",image="k8s.gcr.io/pause:3.1",interface="eth0",name="k8s_POD_dns-deployment-6d486c4fb6-slsq6_openims_3067e0c2-81d0-47a6-bdc3-2a458ca24b3b_4",namespace="openims",pod="dns-deployment-6d486c4fb6-slsq6",pod_name="dns-deployment-6d486c4fb6-slsq6"}',
#       'container_network_transmit_bytes_total{container="POD",container_name="POD",id="/kubepods/besteffort/pod5e0c22bb-f250-4c4f-b2f5-d90e91266229/50f111343380e542220175129525047a784b17efef855af70b0be18ea3588c9e",image="k8s.gcr.io/pause:3.1",interface="eth0",name="k8s_POD_scscf-deployment-56679d845-cnjr7_openims_5e0c22bb-f250-4c4f-b2f5-d90e91266229_4",namespace="openims",pod="scscf-deployment-56679d845-cnjr7",pod_name="scscf-deployment-56679d845-cnjr7"}',
#       'container_network_transmit_bytes_total{container="POD",container_name="POD",id="/kubepods/besteffort/podae951ca5-1dca-4268-af5a-8efc397d5cda/33867386d82e99c8e83c6ff82038de4f25d676c1f351e3457ae819235078e27e",image="k8s.gcr.io/pause:3.1",interface="eth0",name="k8s_POD_icscf-deployment-74f5bb8559-wppfp_openims_ae951ca5-1dca-4268-af5a-8efc397d5cda_5",namespace="openims",pod="icscf-deployment-74f5bb8559-wppfp",pod_name="icscf-deployment-74f5bb8559-wppfp"}',
#       'container_network_transmit_bytes_total{container="POD",container_name="POD",id="/kubepods/besteffort/podd2703fdd-859e-4812-bdcd-282be0659c3e/d8ee6209d0b888c34cdb89bab93974834326915842882831304eba5564db14ec",image="k8s.gcr.io/pause:3.1",interface="eth0",name="k8s_POD_hss-deployment-7b9c7786fd-qgvts_openims_d2703fdd-859e-4812-bdcd-282be0659c3e_17",namespace="openims",pod="hss-deployment-7b9c7786fd-qgvts",pod_name="hss-deployment-7b9c7786fd-qgvts"}',
#       'container_network_transmit_packets_total{container="POD",container_name="POD",id="/kubepods/besteffort/pod1d8d6010-c663-49b3-9000-6483e8d6724c/ed7e2a90b6417b69b10c607d2d2098f042baed5acd1aaccd48761a67fc61157c",image="k8s.gcr.io/pause:3.1",interface="eth0",name="k8s_POD_pcscf-deployment-84649d986b-mk8jp_openims_1d8d6010-c663-49b3-9000-6483e8d6724c_5",namespace="openims",pod="pcscf-deployment-84649d986b-mk8jp",pod_name="pcscf-deployment-84649d986b-mk8jp"}',
#       'container_network_transmit_packets_total{container="POD",container_name="POD",id="/kubepods/besteffort/pod3067e0c2-81d0-47a6-bdc3-2a458ca24b3b/77e9f81f120a4cd1106add9fe98523d1537ce2f576490fb58ba711496f6e0b38",image="k8s.gcr.io/pause:3.1",interface="eth0",name="k8s_POD_dns-deployment-6d486c4fb6-slsq6_openims_3067e0c2-81d0-47a6-bdc3-2a458ca24b3b_4",namespace="openims",pod="dns-deployment-6d486c4fb6-slsq6",pod_name="dns-deployment-6d486c4fb6-slsq6"}',
#       'container_network_transmit_packets_total{container="POD",container_name="POD",id="/kubepods/besteffort/pod5e0c22bb-f250-4c4f-b2f5-d90e91266229/50f111343380e542220175129525047a784b17efef855af70b0be18ea3588c9e",image="k8s.gcr.io/pause:3.1",interface="eth0",name="k8s_POD_scscf-deployment-56679d845-cnjr7_openims_5e0c22bb-f250-4c4f-b2f5-d90e91266229_4",namespace="openims",pod="scscf-deployment-56679d845-cnjr7",pod_name="scscf-deployment-56679d845-cnjr7"}',
#       'container_network_transmit_packets_total{container="POD",container_name="POD",id="/kubepods/besteffort/podae951ca5-1dca-4268-af5a-8efc397d5cda/33867386d82e99c8e83c6ff82038de4f25d676c1f351e3457ae819235078e27e",image="k8s.gcr.io/pause:3.1",interface="eth0",name="k8s_POD_icscf-deployment-74f5bb8559-wppfp_openims_ae951ca5-1dca-4268-af5a-8efc397d5cda_5",namespace="openims",pod="icscf-deployment-74f5bb8559-wppfp",pod_name="icscf-deployment-74f5bb8559-wppfp"}',
#       'container_network_transmit_packets_total{container="POD",container_name="POD",id="/kubepods/besteffort/podd2703fdd-859e-4812-bdcd-282be0659c3e/d8ee6209d0b888c34cdb89bab93974834326915842882831304eba5564db14ec",image="k8s.gcr.io/pause:3.1",interface="eth0",name="k8s_POD_hss-deployment-7b9c7786fd-qgvts_openims_d2703fdd-859e-4812-bdcd-282be0659c3e_17",namespace="openims",pod="hss-deployment-7b9c7786fd-qgvts",pod_name="hss-deployment-7b9c7786fd-qgvts"}']
#            
#            df.drop(df[testDrop],axis=1, inplace=True)    
            
            
            
            ############
        
     