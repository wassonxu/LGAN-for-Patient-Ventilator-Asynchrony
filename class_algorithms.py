# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 11:21:11 2021

@author: Administrator
"""

#       dense_classifier_b() DNN_Forecaster_b LSTM_forecaster_b() LSTM_classifier_b()  


import os
import numpy as np
import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from tensorflow import feature_column
# from tensorflow.keras.layers import layers
from tensorflow import keras
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV,RidgeClassifier

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


class dense_classifier_unb():
    
    def __init__(self):
        pass
    
    def make_one_hot(self, data):
        d=list()
        for i in data:
            if 1 in i:    
                i = 1
                d.append(i)
            else:
                i=0
                d.append(i)
        d = np.array(d).reshape (len(d),1)   # Make the trainy to 0 or 1
        return d 


    def learn(self):
        
        pass
    def split_data_toTrainAndTest(self,dataset,n_input, n_output):
        split = 0.85
        # Seperate the data to Train and Test
        X_train = dataset[:int(split*len(dataset)),1:]
        Y_train = dataset[:int(split*len(dataset)),0]
        X_test = dataset[int(split*len(dataset)):,1:]
        
        Y_test = dataset[int(split*len(dataset)):,0]
        return X_train, Y_train, X_test,Y_test

    def evaluate_model(self,trainx,trainy,testx,testy,n_input):
        #divide the dataset to 0.85/0.15 for training and testing
        x_train=trainx
        y_train=trainy
        x_valid = testx
        y_valid = testy
        # y_train = self.make_one_hot(y_train)
        # y_valid = self.make_one_hot(y_valid)
        
        # train = np.concatenate((x_train,y_train),axis=1)
        # test = np.concatenate((x_valid,y_valid),axis=1)
        # dataset = np.concatenate((train,test),axis = 0)
        
        # x_train=dataset[:int(dataset.shape[0]*0.85),:-1]
        # y_train=dataset[:int(dataset.shape[0]*0.85),-1]
        # x_valid = dataset[int(dataset.shape[0]*0.85):,:-1]
        # y_valid = dataset[int(dataset.shape[0]*0.85):,-1]
        # x_train=np.expand_dims(x_train, axis=-1)
        # x_valid=np.expand_dims(x_valid, axis=-1)
        print(x_train.shape)
        print(y_train.shape)
        print(x_valid.shape)
        print(y_valid.shape)
    
        model = keras.Sequential([
    
            keras.layers.Dense(20, bias_initializer=keras.initializers.glorot_normal()),
            keras.layers.LeakyReLU(),
            keras.layers.Dense(80, bias_initializer=keras.initializers.glorot_normal()),
            keras.layers.LeakyReLU(),
            keras.layers.Dense(100, bias_initializer=keras.initializers.glorot_normal()),
            keras.layers.LeakyReLU(),
            keras.layers.Dense(10, bias_initializer=keras.initializers.glorot_normal()),
            keras.layers.LeakyReLU(),
            keras.layers.Dense(1, activation='sigmoid', bias_initializer=keras.initializers.glorot_normal())
        ])
        # if os._exists('dnn_01.h5'):
        #     model = keras.models.load_model('dnn_01.h5')
    
        op_adam = keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=op_adam,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    
    
        # checkpoint_path = "training_1/cp.ckpt"
        checkpoint_path = 'temp'
        checkpoint_dir = os.path.dirname(checkpoint_path)
    
        # Create checkpoint callback
        cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                         save_weights_only=False,
                                                         monitor='val_accuracy',
                                                         save_best_only=True,
                                                         verbose=1)
        logdir = ".\logdirDNN"
        logdir_dir = os.path.dirname(logdir)
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    
        model.fit(x_train,y_train,
                  validation_split=0.1,
                  # validation_data=(x_valid, y_valid),
                  epochs=30,
                  shuffle=True,
                  batch_size=32,
                  callbacks=[tensorboard_callback,cp_callback],
                  )
        model.save('dense_classifier_unb.h5')
        model.load_weights(checkpoint_path)
        # from keras.models import load_model
        # model = load_model('dense_classifier_unb')
    
        model.summary()
        return model
    
    
    def forecast_one_point(self, model, history, n_input):

    # 	data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
    	# retrieve last observations for input data
    	# reshape into [1, n_input, n]
    	input_x = history.reshape((1, history.shape[0]))
    	# forecast the next week
    	yhat = model.predict_classes(input_x, verbose=0)
    	# we only want the vector forecast
    	yhat = yhat[0]
    	return yhat
    def accuracy(self):
        pass

    def forecast(self, model,testx,testy,n_input):
        predictions = list()
        predictions_origin = list()
        for i in range(len(testx)):
    		# predict the week
            yhat_sequence = self.forecast_one_point(model, testx[i], n_input)
            predictions_origin.append(yhat_sequence)
            if (i%1000) == 0:
                print (i)
            
            
            def threshold(given_value,threshold_value=0.5):
                for ind,val in enumerate(given_value):
                    if val > threshold_value:
                        given_value[ind] = 1
                    else:
                        given_value[ind] = 0
                return given_value
            
    		# store the predictions
    
            # predictions.append(threshold(yhat_sequence.tolist()[:]))
    		# get real observation and add to history for predicting the next week
            # history.append(test[i, :])
        # predictions = array(predictions)
        predictions_origin = array(predictions_origin)
        
        print ('Make one prediction')
        y_pred = list(predictions_origin.reshape(predictions_origin.shape[0]*predictions_origin.shape[1]))
        y_true = list(testy.reshape(predictions_origin.shape[0]*predictions_origin.shape[1])) # 6048 data
        accuracy = accuracy_score(y_true,y_pred)
        tp, fn, fp, tn = confusion_matrix(y_true,y_pred).ravel()
        TPR = tp/(tp+fn)
        TNR = tn/(tn+fp)
        print ('tpr and tnr is', TPR , TNR)
        # accuracy = search_optimal(y_true,y_pred)
        # evaluate predictions days for each week,
        # score, scores = evaluate_forecasts(testy, predictions)
        return accuracy


#%%

import pandas as pd
import numpy
from numpy import array
# from pandas import read_csv
# from sklearn.metrics import mean_squared_error
# from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
# from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix, accuracy_score
# import matplotlib
# import decimal
# from math import sqrt
# from numpy import split
# from numpy import array
# from pandas import read_csv
# from sklearn.metrics import mean_squared_error
# from matplotlib import pyplot
import pandas
# import numpy as np
from collections import Iterable

class LSTM_forecaster_b():
    def __init__(self, gaps_in_the_files = [299.0, 50.0, 50.0, 300.0, 300.0, 299.0, 300.0, 300.0, 299.0, 299.0, 
                     299.0, 299.0, 41.0, 301.0, 339.0, 399.0, 301.0, 301.0, 299.0, 51.0, 301.0, 
                     300.0, 50.0, 300.0, 300.0, 300.0, 49.0, 299.0, 299.0, 300.0, 300.0, 299.0,
                     299.0, 299.0, 299.0, 299.0, 300.0]):
        
        self.gaps_in_the_files = gaps_in_the_files





    def flatten(self,lis):
        "flatten the nested list to one list"
        for item in lis:
            if isinstance(item, Iterable) and not isinstance(item, str):
                for x in self.flatten(item):
                    yield x
            else:        
                yield item
                              
    #calculate the data that should be deleted if we want to make the whole data into one file
    #According to the n_input and n_output
    def delete_the_useless_data(self, dataset,n_input,n_output):
        # total deleted data points would be (Number of Files) * (n_input +n_output - 1)
        total = n_input +n_output
        d=0
        new_gaps = list()
        for i in self.gaps_in_the_files:
            d+=i
            new_gaps.append(d)
            
        Should_deleted_spot = []
        for m in range(1,total):
            Should_deleted_spot.append(np.ndarray.tolist(np.array(new_gaps) - m))
        
        Should_deleted_spot = list(self.flatten(Should_deleted_spot))
        
        Should_deleted_spot = np.array(Should_deleted_spot).astype(int)
        Should_deleted_spot=sorted(Should_deleted_spot, reverse=True) 
        for n in Should_deleted_spot:
            del dataset[n]     # there are 4961 left 
        np.random.shuffle(dataset)
        
            
        dataset = np.array(dataset)
        # X_train = dataset[:,:,1:]
        # y_train = dataset[:,:,0]
         
        
        train = np.array(dataset[:int(len(dataset)*0.85)])
        trainx = train[:,:n_input,:]
        trainy = train[:,n_input:total,0]
        
        
    #balance the train data    
        from imblearn.over_sampling import SMOTE
        from collections import Counter
        x = trainx.reshape(len(trainx),trainx.shape[1]*trainx.shape[2])
        oversample = SMOTE()
        X, trainy = oversample.fit_resample(x, trainy)
        counter = Counter(trainy)
        trainx = X.reshape(len(X),trainx.shape[1],trainx.shape[2])
        
        
        
        test = np.array(dataset[:int(len(dataset)*0.15)])
        testx = test[:,:n_input,:]
        testy = test[:,n_input:total,0]
        
        
        
        
     #balance the test data   
        x = testx.reshape(len(testx),testx.shape[1]*testx.shape[2])
        oversample = SMOTE()
        X, testy = oversample.fit_resample(x, testy)
        counter = Counter(testy)
        testx = X.reshape(len(X),testx.shape[1],testx.shape[2])
     
        
        trainy = trainy.reshape(len(trainy),1)
        testy =testy.reshape(len(testy),1)
        
        return trainx, trainy,testx, testy
    
    
    
    def split_data_toTrainAndTest(self, dataset,n_input,n_output):
        j=0
        New_dataset = []
        dataset = pd.DataFrame(dataset, columns=['Y','I:E ratio','inst_RR','maxF','minF','maxP','PIP','min_pressure','tve:tvi ratio','iTime','eTime','tvi','tve','ipAUC','epAUC'])
        # change the split rate and the data feature can see some changes of results
        dataset = dataset.loc[:,['Y','I:E ratio','inst_RR','maxF','minF','maxP','PIP','iTime','eTime','tvi','tve','ipAUC','epAUC']]
        # dataset = dataset.loc[:,['Y','maxF','minF','maxP','PIP','iTime','eTime','tvi','tve']]
        # dataset = dataset.loc[:,['Y','I:E ratio', 'iTime','eTime','tvi','tve','ipAUC','epAUC']]
        dataset = np.array(dataset)
        for i in dataset:
            dataset_=dataset[j:j+n_input+n_output]
            New_dataset.append(dataset_)
           
            # if j == (len(dataset) - n_input - n_output):
            #     break
            j= j + 1
            # print (j)
            
        trainx,trainy,testx,testy = self.delete_the_useless_data(New_dataset,n_input,n_output)
        return trainx,trainy,testx,testy        
    
    def build_model(self, trainx,trainy, n_input):
     	# prepare data
     	print ('building the model')
     	train_x, train_y = trainx , trainy
     	# define parameters
     	verbose, epochs, batch_size = 1, 10, 32
     	n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
     	# reshape output into [samples, timesteps, features]
     	train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
     	# define model
     	model = Sequential()
     	model.add(LSTM(100, activation='relu', input_shape=(n_timesteps, n_features)))
     	model.add(RepeatVector(n_outputs))
     	model.add(LSTM(100, activation='relu', return_sequences=True))
     	# model.add(TimeDistributed(Dense(100, activation='relu')))
     	# model.add(TimeDistributed(Dense(100, activation='relu')))
     	model.add(TimeDistributed(Dense(1)))
     	model.compile(loss='mse', optimizer='adam')        # mse 81% adam  binary_crossentropy0.5  //  sgd is not working in here
     	# fit network
     	model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
     	print ('model complete')
     	return model
     
    # make a forecast
    def forecast_one_point(self, model, history, n_input):
    
    # 	data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
    	# retrieve last observations for input data
    	# reshape into [1, n_input, n]
    	input_x = history.reshape((1, history.shape[0], history.shape[1]))
    	# forecast the next week
    	yhat = model.predict_classes(input_x, verbose=0)
    	# we only want the vector forecast
    	yhat = yhat[0]
    	return yhat
    
    # # evaluate one or more weekly forecasts against expected values
    # def evaluate_forecasts(self, actual, predicted):
    # 	scores = list()
    # 	# calculate an RMSE score for each day
    # 	for i in range(actual.shape[1]):
    # 		# calculate mse
    # 		mse = mean_squared_error(actual[:, i], predicted[:, i])
    # 		# calculate rmse
    # 		rmse = sqrt(mse)
    # 		# store
    # 		scores.append(rmse)
    # 	# calculate overall RMSE
    # 	s = 0
    # 	for row in range(actual.shape[0]):
    # 		for col in range(actual.shape[1]):
    # 			s += (actual[row, col] - predicted[row, col])**2
    # 	score = sqrt(s / (actual.shape[0] * actual.shape[1]))
    # 	return score, scores
    
    # # summarize scores
    # def summarize_scores(self, name, score, scores):
    # 	s_scores = ', '.join(['%.1f' % s for s in scores])
    # 	print('%s: [%.3f] %s' % (name, score, s_scores))
    
    # evaluate a single model
    def evaluate_model( self, trainx,trainy, testx,testy,n_input):
        model = self.build_model(trainx,trainy, n_input)
            	# history is a list of weekly data
        # history = [x for x in train]
            	# walk-forward validation over each week
        return model
        
    
    def forecast(self, model,testx,testy,n_input):
        predictions = list()
        predictions_origin = list()
        for i in range(len(testx)):
    		# predict the week
            yhat_sequence = self.forecast_one_point(model, testx[i], n_input)
            predictions_origin.append(yhat_sequence)
            
            
            def threshold(given_value,threshold_value=0.5):
                for ind,val in enumerate(given_value):
                    if val > threshold_value:
                        given_value[ind] = 1
                    else:
                        given_value[ind] = 0
                return given_value
            
    		# store the predictions
    
            # predictions.append(threshold(yhat_sequence.tolist()[:]))
    		# get real observation and add to history for predicting the next week
            # history.append(test[i, :])
        # predictions = array(predictions)
        predictions_origin = array(predictions_origin)
        print ('Make one prediction')
        y_pred = list(predictions_origin.reshape(predictions_origin.shape[0]*predictions_origin.shape[1]))
        y_true = list(testy.reshape(predictions_origin.shape[0]*predictions_origin.shape[1])) # 6048 data
        accuracy = accuracy_score(y_true,y_pred)
        # accuracy = search_optimal(y_true,y_pred)
        # evaluate predictions days for each week,
        # score, scores = evaluate_forecasts(testy, predictions)
        return accuracy
    
    def give_combined_data(self, trainx,trainy):
        d=list()
        for i in trainy:
            if 1 in i:    
                i = 1
                d.append(i)
            else:
                i=0
                d.append(i)
        d = np.array(d).reshape (len(d),1)   # Make the trainy to 0 or 1
        
        
        new_data = np.concatenate((d,trainx),axis = 1)  #  new_data = data with label
        
        return new_data

#%%

from statsmodels.tsa.arima_model import ARIMA    
class ARIMA_b ():
    def __init__(self, gaps_in_the_files = [299.0, 50.0, 50.0, 300.0, 300.0, 299.0, 300.0, 300.0, 299.0, 299.0, 
                     299.0, 299.0, 41.0, 301.0, 339.0, 399.0, 301.0, 301.0, 299.0, 51.0, 301.0, 
                     300.0, 50.0, 300.0, 300.0, 300.0, 49.0, 299.0, 299.0, 300.0, 300.0, 299.0,
                     299.0, 299.0, 299.0, 299.0, 300.0]):
        
        self.gaps_in_the_files = gaps_in_the_files





    def flatten(self,lis):
        "flatten the nested list to one list"
        for item in lis:
            if isinstance(item, Iterable) and not isinstance(item, str):
                for x in self.flatten(item):
                    yield x
            else:        
                yield item
                              
    #calculate the data that should be deleted if we want to make the whole data into one file
    #According to the n_input and n_output
    def delete_the_useless_data(self, dataset,n_input,n_output):
        # total deleted data points would be (Number of Files) * (n_input +n_output - 1)
        total = n_input +n_output
        d=0
        new_gaps = list()
        for i in self.gaps_in_the_files:
            d+=i
            new_gaps.append(d)
            
        Should_deleted_spot = []
        for m in range(1,total):
            Should_deleted_spot.append(np.ndarray.tolist(np.array(new_gaps) - m))
        
        Should_deleted_spot = list(self.flatten(Should_deleted_spot))
        
        Should_deleted_spot = np.array(Should_deleted_spot).astype(int)
        Should_deleted_spot=sorted(Should_deleted_spot, reverse=True) 
        for n in Should_deleted_spot:
            del dataset[n]     # there are 4961 left 
        np.random.shuffle(dataset)
        
            
        dataset = np.array(dataset)
        # X_train = dataset[:,:,1:]
        # y_train = dataset[:,:,0]
         
        
        train = np.array(dataset[:int(len(dataset)*0.85)])
        trainx = train[:,:n_input,:]
        trainy = train[:,n_input:total,0]
        
        
    #balance the train data    
        from imblearn.over_sampling import SMOTE
        from collections import Counter
        x = trainx.reshape(len(trainx),trainx.shape[1]*trainx.shape[2])
        oversample = SMOTE()
        X, trainy = oversample.fit_resample(x, trainy)
        counter = Counter(trainy)
        trainx = X.reshape(len(X),trainx.shape[1],trainx.shape[2])
        
        
        
        test = np.array(dataset[:int(len(dataset)*0.15)])
        testx = test[:,:n_input,:]
        testy = test[:,n_input:total,0]
        
        
        
        
     #balance the test data   
        x = testx.reshape(len(testx),testx.shape[1]*testx.shape[2])
        oversample = SMOTE()
        X, testy = oversample.fit_resample(x, testy)
        counter = Counter(testy)
        testx = X.reshape(len(X),testx.shape[1],testx.shape[2])
     
        
        trainy = trainy.reshape(len(trainy),1)
        testy =testy.reshape(len(testy),1)
        
        return trainx, trainy,testx, testy
    
    
    
    def split_data_toTrainAndTest(self, dataset,n_input,n_output):
        j=0
        New_dataset = []
        for i in dataset:
            dataset_=dataset[j:j+n_input+n_output]
            New_dataset.append(dataset_)
           
            # if j == (len(dataset) - n_input - n_output):
            #     break
            j= j + 1
            # print (j)
            
        trainx,trainy,testx,testy = self.delete_the_useless_data(New_dataset,n_input,n_output)
        return trainx,trainy,testx,testy        
    
    def build_model(self, trainx,trainy, n_input):
     	# prepare data
     	print ('building the model')
     	train_x, train_y = trainx , trainy
     	# define parameters
     	verbose, epochs, batch_size = 1, 10, 32
     	n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
     	# reshape output into [samples, timesteps, features]
     	train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
     	# define model
     	model = Sequential()
     	model.add(LSTM(100, activation='relu', input_shape=(n_timesteps, n_features)))
     	model.add(RepeatVector(n_outputs))
     	model.add(LSTM(100, activation='relu', return_sequences=True))
     	model.add(TimeDistributed(Dense(100, activation='relu')))
     	model.add(TimeDistributed(Dense(1)))
     	model.compile(loss='mse', optimizer='adam')        # mse 81% adam  binary_crossentropy0.5  //  sgd is not working in here
     	# fit network
     	model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
     	print ('model complete')
     	return model
     
    # make a forecast
    def forecast_one_point(self, model, history, n_input):
    
    # 	data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
    	# retrieve last observations for input data
    	# reshape into [1, n_input, n]
    	input_x = history.reshape((1, history.shape[0], history.shape[1]))
    	# forecast the next week
    	yhat = model.predict_classes(input_x, verbose=0)
    	# we only want the vector forecast
    	yhat = yhat[0]
    	return yhat
    def evaluate_model( self, trainx,trainy, testx,testy,n_input):
        model = self.build_model(trainx,trainy, n_input)
            	# history is a list of weekly data
        # history = [x for x in train]
            	# walk-forward validation over each week
        return model
        
    
    def forecast(self, model,testx,testy,n_input):
        predictions = list()
        predictions_origin = list()
        for i in range(len(testx)):
    		# predict the week
            yhat_sequence = self.forecast_one_point(model, testx[i], n_input)
            predictions_origin.append(yhat_sequence)
            
            
            def threshold(given_value,threshold_value=0.5):
                for ind,val in enumerate(given_value):
                    if val > threshold_value:
                        given_value[ind] = 1
                    else:
                        given_value[ind] = 0
                return given_value
            
    		# store the predictions
    
            # predictions.append(threshold(yhat_sequence.tolist()[:]))
    		# get real observation and add to history for predicting the next week
            # history.append(test[i, :])
        # predictions = array(predictions)
        predictions_origin = array(predictions_origin)
        print ('Make one prediction')
        y_pred = list(predictions_origin.reshape(predictions_origin.shape[0]*predictions_origin.shape[1]))
        y_true = list(testy.reshape(predictions_origin.shape[0]*predictions_origin.shape[1])) # 6048 data
        accuracy = accuracy_score(y_true,y_pred)
        # accuracy = search_optimal(y_true,y_pred)
        # evaluate predictions days for each week,
        # score, scores = evaluate_forecasts(testy, predictions)
        return accuracy
    
    def give_combined_data(self, trainx,trainy):
        d=list()
        for i in trainy:
            if 1 in i:    
                i = 1
                d.append(i)
            else:
                i=0
                d.append(i)
        d = np.array(d).reshape (len(d),1)   # Make the trainy to 0 or 1
        
        
        new_data = np.concatenate((d,trainx),axis = 1)  #  new_data = data with label
        
        return new_data

    
#%% 
from statsmodels.tsa.api import VAR
from sklearn.metrics import confusion_matrix, accuracy_score

class VARIMA():
    def __init__(self):
        pass
    
    def split_data_toTrainAndTest(self,dataset,n_input, n_output):
        split = 0.95
        dataset = pd.DataFrame(dataset, columns=['Y','I:E ratio','inst_RR','maxF','minF','maxP','PIP','min_pressure','tve:tvi ratio','iTime','eTime','tvi','tve','ipAUC','epAUC'])
        # change the split rate and the data feature can see some changes of results
        # dataset = dataset.loc[:,['Y','I:E ratio','inst_RR','maxF','minF','maxP','PIP','iTime','eTime','tvi','tve','ipAUC','epAUC']]
        dataset = dataset.loc[:,['Y','I:E ratio', 'iTime','eTime','tvi','tve','ipAUC','epAUC']]
        
        # nobs = split * dataset.shape[0]
        df_train, df_test = dataset[0:int(split * dataset.shape[0])], dataset[int(split * dataset.shape[0]):]
        X_test = df_test
        Temp = None
        # Seperate the data to Train and Test
        # the split happened in the evaluate_model function
        return dataset, df_train, X_test,Temp
        
    def evaluate_model(self, dataset, trainy, testx,testy,n_input):
         # Vector autogression  example from https://www.statsmodels.org/dev/vector_ar.html
        df = pd.DataFrame(dataset, columns = ['Y','I:E ratio','inst_RR','maxF','minF','maxP','PIP','min_pressure','tve:tvi ratio','iTime','eTime','tvi','tve','ipAUC','epAUC'])
        # print(df.shape)  # (9719,15)
        # print (df.tail())
        '''
        # Plot  /  Visualize the Time Series
        fig, axes = plt.subplots(nrows=5, ncols=3, dpi=120, figsize=(10,6))
        for i, ax in enumerate(axes.flatten()):
            data = df[df.columns[i]]
            ax.plot(data, color='red', linewidth=1)
            # Decorations
            ax.set_title(df.columns[i])
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            ax.spines["top"].set_alpha(0)
            ax.tick_params(labelsize=6)
        
        plt.tight_layout();
        '''
        
        '''
        from statsmodels.tsa.vector_ar.vecm import coint_johansen
        def cointegration_test(df, alpha=0.05): 
            """Perform Johanson's Cointegration Test and Report Summary"""
            out = coint_johansen(df,-1,5)
            d = {'0.90':0, '0.95':1, '0.99':2}
            traces = out.lr1
            cvts = out.cvt[:, d[str(1-alpha)]]
            def adjust(val, length= 6): return str(val).ljust(length)
        
            # Summary
            print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
            for col, trace, cvt in zip(df.columns, traces, cvts):
                print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)
        cointegration_test(df)
        '''
        
        
        # we remove two of the unuseful features 

        # Check size
        # print(df_train.shape)  # (119, 8)
        # print(df_test.shape)  # (4, 8)
        
        '''
        # How to Select the Order (P) of VAR model
        model = VAR(df_train)
        for i in [1,2,3,4,5,6,7,8,9,10,11,12,13]:
            result = model.fit(i)
            print('Lag Order =', i)
            print('AIC : ', result.aic)
            print('BIC : ', result.bic)
            print('FPE : ', result.fpe)
            print('HQIC: ', result.hqic, '\n')
            
        x = model.select_order(maxlags=12)
        print (x.summary())
        '''
        
        model = VAR(trainy)
        model_fitted = model.fit(n_input)
        # print (model_fitted.summary())
        # Get the lag order
        return model_fitted
        
    def forecast(self, model_fitted,testx,testy,n_input):
        lag_order = model_fitted.k_ar
        print(lag_order)  #> 4
        
        # Input data for forecasting
        
        forecast_value=[]
        for i in range (int(testx.shape[0])):
            if i == int(testx.shape[0]) - lag_order :
                break
            forecast_input = testx.values[i:i+lag_order]
            fc = model_fitted.forecast(y=forecast_input, steps=1)
            # print (forecast_input[:,0])
            forecast_value.append(fc[:,0])
        real_value = testx.values[lag_order:,0]
        
        
        # calculate the accuracy score
        d=[]
        threshold = np.arange(0.3,0.8,0.01)
        accuracy_ = []
        for val in threshold:
            
            for i in forecast_value:
                i = list(i)
                if i[0] > val:
                    d.append(1)
                else:
                    d.append(0)
            
            accuracy = accuracy_score(real_value,d)
            accuracy_.append(accuracy)
            tp, fn, fp, tn = confusion_matrix(real_value,d).ravel()
            TPR = tp/(tp+fn)
            TNR = tn/(tn+fp)
            print ('tp, fn, fp, tn',tp, fn, fp, tn)
            print ('tpr and tnr is', TPR, TNR)
            d=[]
        return (max(accuracy_))

#%%    
class DNN_Forecaster_b():
    def __init__(self, gaps_in_the_files = [299.0, 50.0, 50.0, 300.0, 300.0, 299.0, 300.0, 300.0, 299.0, 299.0, 
                     299.0, 299.0, 41.0, 301.0, 339.0, 399.0, 301.0, 301.0, 299.0, 51.0, 301.0, 
                     300.0, 50.0, 300.0, 300.0, 300.0, 49.0, 299.0, 299.0, 300.0, 300.0, 299.0,
                     299.0, 299.0, 299.0, 299.0, 300.0]):
        
        self.gaps_in_the_files = gaps_in_the_files





    def flatten(self,lis):
        "flatten the nested list to one list"
        for item in lis:
            if isinstance(item, Iterable) and not isinstance(item, str):
                for x in self.flatten(item):
                    yield x
            else:        
                yield item
                              
    #calculate the data that should be deleted if we want to make the whole data into one file
    #According to the n_input and n_output
    def delete_the_useless_data(self, dataset,n_input,n_output):
        # total deleted data points would be (Number of Files) * (n_input +n_output - 1)
        total = n_input +n_output
        d=0
        new_gaps = list()
        for i in self.gaps_in_the_files:
            d+=i
            new_gaps.append(d)
            
        Should_deleted_spot = []
        for m in range(1,total):
            Should_deleted_spot.append(np.ndarray.tolist(np.array(new_gaps) - m))
        
        Should_deleted_spot = list(self.flatten(Should_deleted_spot))
        
        Should_deleted_spot = np.array(Should_deleted_spot).astype(int)
        Should_deleted_spot=sorted(Should_deleted_spot, reverse=True) 
        for n in Should_deleted_spot:
            del dataset[n]     # there are 4961 left 
        np.random.shuffle(dataset)
        
            
        dataset = np.array(dataset)
        # X_train = dataset[:,:,1:]
        # y_train = dataset[:,:,0]
         
        
        train = np.array(dataset[:int(len(dataset)*0.85)])
        trainx = train[:,:n_input,:]
        trainy = train[:,n_input:total,0]
        
        
    #balance the train data    
        from imblearn.over_sampling import SMOTE
        from collections import Counter
        x = trainx.reshape(len(trainx),trainx.shape[1]*trainx.shape[2])
        oversample = SMOTE()
        X, trainy = oversample.fit_resample(x, trainy)
        counter = Counter(trainy)
        trainx = X.reshape(len(X),trainx.shape[1],trainx.shape[2])
        
        
        
        test = np.array(dataset[:int(len(dataset)*0.15)])
        testx = test[:,:n_input,:]
        testy = test[:,n_input:total,0]
        
        
        
        
     #balance the test data   
        x = testx.reshape(len(testx),testx.shape[1]*testx.shape[2])
        oversample = SMOTE()
        X, testy = oversample.fit_resample(x, testy)
        counter = Counter(testy)
        testx = X.reshape(len(X),testx.shape[1],testx.shape[2])
     
        
        trainy = trainy.reshape(len(trainy),1)
        testy =testy.reshape(len(testy),1)
        
        return trainx, trainy,testx, testy
    
    
    
    def split_data_toTrainAndTest(self, dataset,n_input,n_output):
        j=0
        New_dataset = []
        dataset = pd.DataFrame(dataset, columns=['Y','I:E ratio','inst_RR','maxF','minF','maxP','PIP','min_pressure','tve:tvi ratio','iTime','eTime','tvi','tve','ipAUC','epAUC'])
        # change the split rate and the data feature can see some changes of results
        # dataset = dataset.loc[:,['Y','I:E ratio','inst_RR','maxF','minF','maxP','PIP','iTime','eTime','tvi','tve','ipAUC','epAUC']]
        # dataset = dataset.loc[:,['Y','maxF','minF','maxP','PIP','iTime','eTime','tvi','tve']]
        dataset = dataset.loc[:,['Y','I:E ratio','inst_RR','tve:tvi ratio','iTime','eTime','tvi','tve']]
        dataset = np.array(dataset)
        for i in dataset:
            dataset_=dataset[j:j+n_input+n_output]
            New_dataset.append(dataset_)
           
            # if j == (len(dataset) - n_input - n_output):
            #     break
            j= j + 1
            # print (j)
            
        trainx,trainy,testx,testy = self.delete_the_useless_data(New_dataset,n_input,n_output)
        return trainx,trainy,testx,testy        
    
    def build_model(self, trainx,trainy, n_input):
     	# prepare data
     	print ('building the model')
     	train_x, train_y = trainx , trainy
     	# define parameters
     	verbose, epochs, batch_size = 1, 10, 32
     	n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
     	# reshape output into [samples, timesteps, features]
     	train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
     	# define model
     	model = keras.Sequential([
    
            keras.layers.Dense(20, bias_initializer=keras.initializers.glorot_normal()),
            keras.layers.LeakyReLU(),
            keras.layers.Dense(80, bias_initializer=keras.initializers.glorot_normal()),
            keras.layers.LeakyReLU(),
            keras.layers.Dense(100, bias_initializer=keras.initializers.glorot_normal()),
            keras.layers.LeakyReLU(),
            keras.layers.Dense(10, bias_initializer=keras.initializers.glorot_normal()),
            keras.layers.LeakyReLU(),
            keras.layers.Dense(1, activation='sigmoid', bias_initializer=keras.initializers.glorot_normal())
        ])
        # if os._exists('dnn_01.h5'):
        #     model = keras.models.load_model('dnn_01.h5')
    
     	op_adam = keras.optimizers.Adam(learning_rate=0.0001)
     	model.compile(optimizer=op_adam,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    

    
     	model.fit(train_x,trainy,
                  validation_split=0.9,
                  # validation_data=(x_valid, y_valid),
                  epochs=5,
                  shuffle=True,
                  batch_size=32,
                  # callbacks=[tensorboard_callback],
                  )
     	print ('model complete')
     	return model
     
    # make a forecast
    def forecast_one_point(self, model, history, n_input):
    
    # 	data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
    	# retrieve last observations for input data
    	# reshape into [1, n_input, n]
    	input_x = history.reshape((1, history.shape[0], history.shape[1]))
    	# forecast the next week
    	yhat = model.predict_classes(input_x, verbose=0)
    	# we only want the vector forecast
    	yhat = yhat[0]
    	return yhat
    
    # # evaluate one or more weekly forecasts against expected values
    # def evaluate_forecasts(self, actual, predicted):
    # 	scores = list()
    # 	# calculate an RMSE score for each day
    # 	for i in range(actual.shape[1]):
    # 		# calculate mse
    # 		mse = mean_squared_error(actual[:, i], predicted[:, i])
    # 		# calculate rmse
    # 		rmse = sqrt(mse)
    # 		# store
    # 		scores.append(rmse)
    # 	# calculate overall RMSE
    # 	s = 0
    # 	for row in range(actual.shape[0]):
    # 		for col in range(actual.shape[1]):
    # 			s += (actual[row, col] - predicted[row, col])**2
    # 	score = sqrt(s / (actual.shape[0] * actual.shape[1]))
    # 	return score, scores
    
    # # summarize scores
    # def summarize_scores(self, name, score, scores):
    # 	s_scores = ', '.join(['%.1f' % s for s in scores])
    # 	print('%s: [%.3f] %s' % (name, score, s_scores))
    
    # evaluate a single model
    def evaluate_model( self, trainx,trainy, testx,testy,n_input):
        model = self.build_model(trainx,trainy, n_input)
            	# history is a list of weekly data
        # history = [x for x in train]
            	# walk-forward validation over each week
        return model
        
    
    def forecast(self, model,testx,testy,n_input):
        predictions = list()
        predictions_origin = list()
        for i in range(len(testx)):
    		# predict the week
            yhat_sequence = self.forecast_one_point(model, testx[i], n_input)
            predictions_origin.append(yhat_sequence)
            
            
            def threshold(given_value,threshold_value=0.5):
                for ind,val in enumerate(given_value):
                    if val > threshold_value:
                        given_value[ind] = 1
                    else:
                        given_value[ind] = 0
                return given_value
            
    		# store the predictions
    
            # predictions.append(threshold(yhat_sequence.tolist()[:]))
    		# get real observation and add to history for predicting the next week
            # history.append(test[i, :])
        # predictions = array(predictions)
        predictions_origin = array(predictions_origin)
        print ('Make one prediction')
        y_pred = list(predictions_origin.reshape(predictions_origin.shape[0]*predictions_origin.shape[1]))
        y_true = list(testy.reshape(predictions_origin.shape[0]*predictions_origin.shape[1])) # 6048 data
        accuracy = accuracy_score(y_true,y_pred)
        # accuracy = search_optimal(y_true,y_pred)
        # evaluate predictions days for each week,
        # score, scores = evaluate_forecasts(testy, predictions)
        return accuracy
    
    def give_combined_data(self, trainx,trainy):
        d=list()
        for i in trainy:
            if 1 in i:    
                i = 1
                d.append(i)
            else:
                i=0
                d.append(i)
        d = np.array(d).reshape (len(d),1)   # Make the trainy to 0 or 1
        
        
        new_data = np.concatenate((d,trainx),axis = 1)  #  new_data = data with label
        
        return new_data

#%%
from imblearn.over_sampling import SMOTE
from collections import Counter
from tensorflow.keras import activations
import random

class dense_classifier_b():
    
    def __init__(self):
        pass
    
    def make_one_hot(self, data):
        d=list()
        for i in data:
            if 1 in i:    
                i = 1
                d.append(i)
            else:
                i=0
                d.append(i)
        d = np.array(d).reshape (len(d),1)   # Make the trainy to 0 or 1
        return d 


    def learn(self):
        
        pass
    def split_data_toTrainAndTest(self,dataset,n_input, n_output):
        split = 0.9
        # Seperate the data to Train and Test
        np.random.shuffle(dataset)

        X_train = dataset[:int(split*len(dataset)),1:]
        Y_train = dataset[:int(split*len(dataset)),0]
        oversample = SMOTE()
        X_train, Y_train = oversample.fit_resample(X_train,Y_train)
        X_test = dataset[int(split*len(dataset)):,1:]
        
        Y_test = dataset[int(split*len(dataset)):,0]
        return X_train, Y_train, X_test,Y_test

    def evaluate_model(self,trainx,trainy,testx,testy,n_input):
        #divide the dataset to 0.85/0.15 for training and testing
        x_train=trainx
        y_train=trainy
        x_valid = testx
        y_valid = testy
        # y_train = self.make_one_hot(y_train)
        # y_valid = self.make_one_hot(y_valid)
        
        # train = np.concatenate((x_train,y_train),axis=1)
        # test = np.concatenate((x_valid,y_valid),axis=1)
        # dataset = np.concatenate((train,test),axis = 0)
        
        # x_train=dataset[:int(dataset.shape[0]*0.85),:-1]
        # y_train=dataset[:int(dataset.shape[0]*0.85),-1]
        # x_valid = dataset[int(dataset.shape[0]*0.85):,:-1]
        # y_valid = dataset[int(dataset.shape[0]*0.85):,-1]
        # x_train=np.expand_dims(x_train, axis=-1)
        # x_valid=np.expand_dims(x_valid, axis=-1)
        print(x_train.shape)
        print(y_train.shape)
        print(x_valid.shape)
        print(y_valid.shape)
    
        model = keras.Sequential([
    
            keras.layers.Dense(200, activation = activations.sigmoid, bias_initializer=keras.initializers.glorot_normal()),
            # keras.layers.LeakyReLU(),
            # keras.layers.Dense(100, activation = activations.sigmoid, bias_initializer=keras.initializers.glorot_normal()),
            # keras.layers.Dense(100, activation = activations.sigmoid, bias_initializer=keras.initializers.glorot_normal()),
            # keras.layers.Dense(100, activation = activations.sigmoid, bias_initializer=keras.initializers.glorot_normal()),
            keras.layers.Dense(200, activation = activations.sigmoid, bias_initializer=keras.initializers.glorot_normal()),
            keras.layers.Dense(200, activation = activations.sigmoid, bias_initializer=keras.initializers.glorot_normal()),
            # keras.layers.LeakyReLU(),
            # keras.layers.Dense(100, bias_initializer=keras.initializers.glorot_normal()),
            # keras.layers.LeakyReLU(),
            # keras.layers.Dense(10, bias_initializer=keras.initializers.glorot_normal()),
            # keras.layers.LeakyReLU(),
            keras.layers.Dense(1, activation='sigmoid', bias_initializer=keras.initializers.glorot_normal())
        ])
        # if os._exists('dnn_01.h5'):
        #     model = keras.models.load_model('dnn_01.h5')
    
        op_adam = keras.optimizers.Adam(learning_rate=0.005)
        model.compile(optimizer=op_adam,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    
    
        # checkpoint_path = "training_1/cp.ckpt"
        checkpoint_path = 'temp'
        checkpoint_dir = os.path.dirname(checkpoint_path)
    
        # Create checkpoint callback
        cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                         save_weights_only=True,
                                                         monitor='val_accuracy',
                                                         save_best_only=True,
                                                         verbose=1)
        logdir = ".\logdirDNN"
        logdir_dir = os.path.dirname(logdir)
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    
        history = model.fit(x_train,y_train,
                  validation_split=0.1,
                  # validation_data=(x_valid, y_valid),
                  epochs=40,
                  shuffle=True,
                  batch_size=32,
                  callbacks=[cp_callback],
                  )
        # print(history.history.keys())
        # summarize history for accuracy
        import matplotlib.pyplot as plt
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # model.save('dense_classifier_b.h5')
        # model.load_weights(checkpoint_path)
        # from keras.models import load_model
        # model = load_model('dense_classifier_unb')
    
        return model
    
    
    def forecast_one_point(self, model, history, n_input):

    # 	data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
    	# retrieve last observations for input data
    	# reshape into [1, n_input, n]
    	input_x = history.reshape((1, history.shape[0]))
    	# forecast the next week
    	yhat = model.predict_classes(input_x, verbose=0)
    	# we only want the vector forecast
    	yhat = yhat[0]
    	return yhat
    def accuracy(self):
        pass

    def forecast(self, model,testx,testy,n_input):
        
        
        # prediction of whold
        predictions_origin = model.predict_classes(testx,  verbose=0)
        
        
        
        
        
      #   predictions = list()
      #   predictions_origin = list()
      #   for i in range(len(testx)):
    		# # predict the week
      #       yhat_sequence = self.forecast_one_point(model, testx[i], n_input)
      #       predictions_origin.append(yhat_sequence)
      #       if (i% 1000) == 0:
      #           print (i)
            
            
      #       def threshold(given_value,threshold_value=0.5):
      #           for ind,val in enumerate(given_value):
      #               if val > threshold_value:
      #                   given_value[ind] = 1
      #               else:
      #                   given_value[ind] = 0
      #           return given_value
            
    		# store the predictions
    
            # predictions.append(threshold(yhat_sequence.tolist()[:]))
    		# get real observation and add to history for predicting the next week
            # history.append(test[i, :])
        # predictions = array(predictions)
        # predictions_origin = array(predictions_origin)
        print ('Make one prediction')
        y_pred = list(predictions_origin.reshape(predictions_origin.shape[0]*predictions_origin.shape[1]))
        y_true = list(testy.reshape(predictions_origin.shape[0]*predictions_origin.shape[1])) # 6048 data
        accuracy = accuracy_score(y_true,y_pred)
        tp, fn, fp, tn = confusion_matrix(y_true,y_pred).ravel()
        TPR = tp/(tp+fn)
        TNR = tn/(tn+fp)
        print ('tpr and tnr is', TPR , TNR)
        # accuracy = search_optimal(y_true,y_pred)
        # evaluate predictions days for each week,
        # score, scores = evaluate_forecasts(testy, predictions)
        return accuracy

#%%
# from keras.layers import ConvLSTM2D
class LSTM_classifier_b():
    def __init__(self):
        pass
    
    def make_one_hot(self, data):
        d=list()
        for i in data:
            if 1 in i:    
                i = 1
                d.append(i)
            else:
                i=0
                d.append(i)
        d = np.array(d).reshape (len(d),1)   # Make the trainy to 0 or 1
        return d 


    def learn(self):
        
        pass
    def split_data_toTrainAndTest(self,dataset,n_input, n_output):
        split = 0.85
        np.random.shuffle(dataset)
        # Seperate the data to Train and Test
        X_train = dataset[:int(split*len(dataset)),1:]
        Y_train = dataset[:int(split*len(dataset)),0]
        # oversample = SMOTE()
        from imblearn.over_sampling import SVMSMOTE
        oversample = SVMSMOTE()
        X_train, Y_train = oversample.fit_resample(X_train,Y_train)
        X_test = dataset[int(split*len(dataset)):,1:]
        
        Y_test = dataset[int(split*len(dataset)):,0]
        X_test, Y_test = oversample.fit_resample(X_test,Y_test)
        
        
        
        from sklearn.tree import DecisionTreeClassifier
        from imblearn.pipeline import Pipeline
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler
        from sklearn.model_selection import RepeatedStratifiedKFold
        from sklearn.model_selection import cross_val_score
        from numpy import mean
        k_values = [1, 2, 3, 4, 5, 6, 7]
        for k in k_values:
        # define pipeline
            model = DecisionTreeClassifier()
            over = SMOTE(sampling_strategy=0.1, k_neighbors=k)
            under = RandomUnderSampler(sampling_strategy=0.5)
            steps = [('over', over), ('under', under), ('model', model)]
            pipeline = Pipeline(steps=steps)
            # evaluate pipeline
            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
            scores = cross_val_score(pipeline, X_train,Y_train, scoring='roc_auc', cv=cv, n_jobs=-1)
            score = mean(scores)
            
            print('> k=%d, Mean ROC AUC: %.3f' % (k, score))
            
        
        return X_train, Y_train, X_test,Y_test

    def evaluate_model(self,trainx,trainy,testx,testy,n_input):

     	# prepare data
     	print ('building the model')
     	train_x, train_y = trainx , trainy
     	verbose, epochs, batch_size = 1, 30, 8
     	train_x = train_x.reshape(train_x.shape[0],train_x.shape[1],1)
     	train_y = train_y.reshape(train_y.shape[0],1)
    
     	n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
     	# reshape output into [samples, timesteps, features]
     	train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
     	# define model
     	model = Sequential()
     	# model.add(ConvLSTM2D(64,(1,3), activation='relu', input_shape=(n_timesteps, n_features)))
     	model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
     	model.add(RepeatVector(n_outputs))
     	model.add(LSTM(200, activation='relu', return_sequences=True))
     	# model.add(TimeDistributed(Dense(100, activation='relu')))
     	model.add(TimeDistributed(Dense(1)))
     	model.compile(loss='mse', optimizer='adam')
     	# model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')
        # model.compile(loss='binary_crossentropy', metrics=['accuracy'])                      
     	# fit network
     	history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
 
     	return model
    
    
    def forecast_one_point(self, model, history, n_input):

    # 	data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
    	# retrieve last observations for input data
    	# reshape into [1, n_input, n]
    	input_x = history.reshape((1, history.shape[0],1))
    	# forecast the next week
    	yhat = model.predict_classes(input_x, verbose=0)
    	# we only want the vector forecast
    	yhat = yhat[0]
    	return yhat
    def accuracy(self):
        pass

    def forecast(self, model,testx,testy,n_input):
        
        
        # predictions_origin = model.predict_classes(testx, verbose=0)
        
        
        predictions_origin = list()
        for i in range(len(testx)):
    		 # predict the week
             yhat_sequence = self.forecast_one_point(model, testx[i], n_input)
             predictions_origin.append(yhat_sequence)
             if (i% 1000) == 0:
                 print (i)
            
            
             def threshold(given_value,threshold_value=0.5):
                 for ind,val in enumerate(given_value):
                     if val > threshold_value:
                         given_value[ind] = 1
                     else:
                         given_value[ind] = 0
                 return given_value
            
    		 # store the predictions
    
             # predictions.append(threshold(yhat_sequence.tolist()[:]))
    		 # get real observation and add to history for predicting the next week
             # history.append(test[i, :])
         # predictions = array(predictions)
        predictions_origin = array(predictions_origin)
        print ('Make one prediction')
        y_pred = list(predictions_origin.reshape(predictions_origin.shape[0]*predictions_origin.shape[1]))
        y_true = list(testy.reshape(predictions_origin.shape[0]*predictions_origin.shape[1])) # 6048 data
        accuracy = accuracy_score(y_true,y_pred)
        tp, fn, fp, tn = confusion_matrix(y_true,y_pred).ravel()
        TPR = tp/(tp+fn)
        TNR = tn/(tn+fp)
        print ('tpr and tnr is', TPR , TNR)
        # accuracy = search_optimal(y_true,y_pred)
        # evaluate predictions days for each week,
        # score, scores = evaluate_forecasts(testy, predictions)
        return accuracy
    
class KernelLogisticRegression():
    
    def __init__(self):
        pass
    
    def make_one_hot(self, data):
        d=list()
        for i in data:
            if 1 in i:    
                i = 1
                d.append(i)
            else:
                i=0
                d.append(i)
        d = np.array(d).reshape (len(d),1)   # Make the trainy to 0 or 1
        return d 


    def learn(self):
        
        pass
    def split_data_toTrainAndTest(self,dataset,n_input, n_output):
        split = 0.90
        np.random.shuffle(dataset)
        # Seperate the data to Train and Test
        X_train = dataset[:int(split*len(dataset)),1:]
        Y_train = dataset[:int(split*len(dataset)),0]
        # oversample = SMOTE()
        from imblearn.over_sampling import SVMSMOTE
        oversample = SVMSMOTE()
        X_train, Y_train = oversample.fit_resample(X_train,Y_train)
        X_test = dataset[int(split*len(dataset)):,1:]
        
        Y_test = dataset[int(split*len(dataset)):,0]
        X_test, Y_test = oversample.fit_resample(X_test,Y_test)
        
        
        
        # from sklearn.tree import DecisionTreeClassifier
        # from imblearn.pipeline import Pipeline
        # from imblearn.over_sampling import SMOTE
        # from imblearn.under_sampling import RandomUnderSampler
        # from sklearn.model_selection import RepeatedStratifiedKFold
        # from sklearn.model_selection import cross_val_score
        # from numpy import mean
        # k_values = [1, 2, 3, 4, 5, 6, 7]
        # for k in k_values:
        # # define pipeline
        #     model = DecisionTreeClassifier()
        #     over = SMOTE(sampling_strategy=0.1, k_neighbors=k)
        #     under = RandomUnderSampler(sampling_strategy=0.5)
        #     steps = [('over', over), ('under', under), ('model', model)]
        #     pipeline = Pipeline(steps=steps)
        #     # evaluate pipeline
        #     cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        #     scores = cross_val_score(pipeline, X_train,Y_train, scoring='roc_auc', cv=cv, n_jobs=-1)
        #     score = mean(scores)
            
        #     print('> k=%d, Mean ROC AUC: %.3f' % (k, score))
            
        
        return X_train, Y_train, X_test,Y_test
    def evaluate_model(self,trainx,trainy,testx,testy,n_input):

     	# prepare data
        print ('building the model')
        model = LogisticRegression(random_state=0).fit(trainx, trainy)
 
        return model
    def forecast(self, model,testx,testy,n_input):
        
        
        # prediction of whold
        predictions_origin = model.predict(testx)
        
        
        print ('Make one prediction')
        y_pred = list(predictions_origin.reshape(predictions_origin.shape[0]))
        y_true = list(testy.reshape(predictions_origin.shape[0])) # 6048 data
        accuracy = accuracy_score(y_true,y_pred)
        tp, fn, fp, tn = confusion_matrix(y_true,y_pred).ravel()
        TPR = tp/(tp+fn)
        TNR = tn/(tn+fp)
        print ('tpr and tnr is', TPR , TNR)
        # accuracy = search_optimal(y_true,y_pred)
        # evaluate predictions days for each week,
        # score, scores = evaluate_forecasts(testy, predictions)
        return accuracy
# if __name__ == '__main__':
#     # df = pd.read_csv('dataset\weather-dataset-rattle-package\weatherAUS.csv')
#     # train, test = train_test_split(df, test_size=0.2, random_state=1)
#     # train, valid = train_test_split(train,test_size=0.2, random_state=1)
#     # y_train=np.array(train.iloc[:, [7]])
#     # x_train=np.array(train.iloc[:,[0,1,2,3,4,5,6]])
#     # np.save('x_train.npy', x_train)
#     # np.save('y_train.npy', y_train)
#     # y_valid = np.array(valid.iloc[:, [7]])
#     # x_valid = np.array(valid.iloc[:,[0,1,2,3,4,5,6]])
#     # np.save('x_valid.npy', x_valid)
#     # np.save('y_valid.npy', y_valid)
#     # y_test= np.array(test.iloc[:, [7]])
#     # x_test = np.array(test.iloc[:,[0,1,2,3,4,5,6]])
#     # np.save('x_test.npy', x_test)
#     # np.save('y_test.npy', y_test)
#     dense_classifier().run()

