# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 11:07:47 2021

@author: Administrator
"""


import pandas as pd
import numpy as np
import class_algorithms as algs
from utilities.utilities import Change_array_to_list


def give_combined_data(trainx,trainy):
    # combine the trainx and trainy to one file with the trainy at the first column
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

def one_hot(trainy):
    d=list()
    for i in trainy:
        if 1 in i:    
            i = 1
            d.append(i)
        else:
            i=0
            d.append(i)
    d = np.array(d).reshape (len(d),1)   # Make the trainy to 0 or 1
    return d    


from collections import Counter
def get_BSA(trainy):
    d = list ()
    for i in trainy:
        if i[1] == 1:
            i = 1
            d.append(i)
        else:
            i = 0
            d.append(i)
    d = np.array(d).reshape(len(d),1)
    return d

def get_BSA_and_normal (trainy):
    d = list ()
    normal = '[0 0 0 0 0]'
    bsa = '[0 1 0 0 0]'
    for i in trainy:
        if i == bsa:
            i = 1
            d.append(i)
        else:
            i = 0
            d.append(i)
    d = np.array(d).reshape(len(d),1)
    return d

# trainy_bsa_list=trainy_bsa.reshape(len(trainy_bsa),).tolist()
# counter = Counter(trainy_bsa_list)

def get_DTA_and_normal(trainy):
    d = list ()
    normal = '[0 0 0 0 0]'
    bsa = '[1 0 0 0 0]'
    for i in trainy:
        if i == bsa:
            i = 1
            d.append(i)
        else:
            i = 0
            d.append(i)
    d = np.array(d).reshape(len(d),1)
    return d  

#%%
# trainX = pd.read_csv('train_all.csv')
# trainY = np.load('trainy_all.npy')


# here is seperate the BSA and The other

trainX = pd.read_csv('Seperated_data\ALL_data_trainX')
trainY = np.load('Seperated_data\ALL_data_trainY.npy')


retro_1 = pd.read_csv('retro_data/train_all_retro_1_after_check.csv')
retro_2 = pd.read_csv('retro_data/train_all_retro_2_after_check.csv')
retro_3 = pd.read_csv('retro_data/train_all_retro_3_after_check.csv')



List_trainY = Change_array_to_list(trainY)
trainy = get_BSA_and_normal(List_trainY)
# counter = Counter(trainy.reshape(len(trainy),).tolist())

''' # here is the comparation of  normal and BSA 
trainX = pd.read_csv('Seperated_data\ALL_Normal_trainX')
trainY = np.load('Seperated_data\ALL_Normal_trainY.npy')

trainX_BSA = pd.read_csv('Seperated_data\ALL_BSA_trainX')
trainY_BSA = np.load('Seperated_data\ALL_BSA_trainY.npy')

trainX = pd.concat([trainX, trainX_BSA])
trainY = np.concatenate ((trainY, trainY_BSA),axis=0)

'''



numruns = 1
n_input = 3
n_output =1

# get the useful feature from the trainX
# Useful_data = retro_1.loc[:, ['I:E ratio','inst_RR','maxF','minF','maxP','PIP','min_pressure','tve:tvi ratio','iTime','eTime','tvi','tve','ipAUC','epAUC','-1_I:E ratio','-1_inst_RR','-1_maxF','-1_minF','-1_maxP','-1_PIP','-1_min_pressure','-1_tve:tvi ratio','-1_iTime','-1_eTime','-1_tvi','-1_tve','-1_ipAUC','-1_epAUC']]
# Useful_data = retro_1.loc[:, ['I:E ratio','inst_RR','tve:tvi ratio','iTime','eTime','tvi','tve','ipAUC','-1_I:E ratio','-1_inst_RR','-1_tve:tvi ratio','-1_iTime','-1_eTime','-1_tvi','-1_tve','-1_ipAUC']]
# Useful_data = trainX.loc[:, ['I:E ratio','inst_RR','tve:tvi ratio','iTime','eTime','tvi','tve']]
Useful_data = retro_1.loc[:, ['I:E ratio','inst_RR','maxF','minF','maxP','PIP','min_pressure','tve:tvi ratio','iTime','eTime','tvi','tve','ipAUC','epAUC']]
# Useful_data = trainX.loc[:, ['I:E ratio','inst_RR','tve:tvi ratio','eTime','tve']]
useful_data = np.array(Useful_data)

# normalize the data and see what is going on
# import matplotlib.pyplot as plt
# plt.plot(Useful_data.loc[:,['iTime']])
# plt.show()
# audio /= np.max(np.abs(audio),axis=0)

useful_data = np.nan_to_num(useful_data)

# standardize the array
useful_data = (useful_data - np.mean(useful_data, axis=0)) / np.std(useful_data, axis=0)

#get the trainy of one-hot
# trainy = one_hot(trainY)

# trainy_bsa = get_BSA(trainY)  # get BSA data
# get combined dataset
dataset = give_combined_data(useful_data, trainY)
# dataset = dataset[600:1040,:]





regressionalgs = {
#        'Random': algs.Regressor,
        # 'Mean': algs.MeanPredictor,
#        'FSLinearRegression': algs.FSLinearRegression,
        # 'RidgeLinearRegression': algs.RidgeLinearRegression,
        # 'KernelLinearRegression': algs.KernelLinearRegression,
        # 'LinearRegression': algs.LinearRegression,
        # 'MPLinearRegression': algs.MPLinearRegression,
        # 'Dense_Regression': algs.dense_classifier_unb,
        
        # 'LSTM_classifier',algs.LSTM_forecaster_b,
        # 'VARIMA': algs.VARIMA,
        # 'Kernel Logistic Regression': algs.KernelLogisticRegression,
        'Dense_Regression_balanced': algs.dense_classifier_b,
        # 'LSTM_classifier': algs.LSTM_classifier_b,
        #'CNN_classifier':algs.CNN_b
        # 'ARIMA_forecaster': algs.ARIMA_b,
        # 'DNN':algs.DNN_Forecaster_b
        # 'GAN': algs.GAN,
#        'BGD': algs.BGD,
#        'MomentumSGD':algs.MomentumSGD,
#        'AdamSGD':algs.AdamSGD,
    }



for r in range(numruns):
        # trainset, testset = dtl.load_ctscan(trainsize,testsize)
        

        for learnername, Learner in regressionalgs.items():
            # params = parameters.get(learnername, [ None ])
            print ('Running learner = ' + learnername)
                # Train model
            trainx, trainy, testx, testy = Learner().split_data_toTrainAndTest(dataset,n_input, n_output)
            print(('Running on train={0} and test={1} samples for run {2}').format(trainx.shape[0], testx.shape[0], r))
            model = Learner().evaluate_model(trainx,trainy,testx,testy,n_input)
            accuracy = Learner().forecast(model,testx,testy,n_input)
            
            print (accuracy)
# #                learner.learn(trainset[0], trainset[1])
#                 # Test model
#                 predictions = learner.predict(testset[0])
#                 error = geterror(testset[1], predictions)
#                 print ('Error for ' + learnername + ': ' + str(error))
#                 errors[learnername][p, r] = error


# x_train = useful_data.reshape((useful_data.shape[0], useful_data.shape[1], 1))
# num_classes = len(np.unique(trainy))
# import matplotlib.pyplot as plt
# plt.matshow(Useful_data.corr())
# plt.show()
'''
#%%
# see the trend of the value and look for the stationarybility
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
# Draw Plot
def plot_df(df, x, y, title="", xlabel='Breath time points', ylabel='Value of', dpi=100):
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()

Useful_data_example = Useful_data.loc[:,['iTime','eTime','tvi']]

for i in Useful_data_example.columns:
    for j, k in [[0,299],[399,699],[699,999]]:
        plot_df(Useful_data_example, x=Useful_data_example.index[j:k], y= Useful_data_example.loc[j:k-1,i], ylabel = 'Value of '+ i, title='The trend of the '+ i +' in single patient')   

#%% check the p-value by adfuller
Useful_data = trainX.loc[:, ['I:E ratio','inst_RR','maxF','minF','maxP','PIP','tve:tvi ratio','iTime','eTime','tvi','tve','ipAUC','epAUC']]
for i in Useful_data.columns:
    print (i)
    result = adfuller( Useful_data.loc[:,i], autolag='AIC')
    print(f'p-value: {result[1]}')

#%% test to know if one time series is helpful in forecasting another?
from statsmodels.tsa.stattools import grangercausalitytests
df = pd.DataFrame(dataset, columns = ['Y','I:E ratio','inst_RR','maxF','minF','maxP','PIP','min_pressure','tve:tvi ratio','iTime','eTime','tvi','tve','ipAUC','epAUC'])

grangercausalitytests(df[['Y', 'min_pressure']], maxlag=5)
for i in df.columns:
    print ("PROCESSING",i)
    grangercausalitytests(df[['Y', i]], maxlag=1)


#%% arima 
# Import data
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/austa.csv')
from statsmodels.tsa.arima_model import ARIMA


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf

# Create Training and Test
train = df.value[:20]
test = df.value[20:]
# Build Model
model = ARIMA(train, order=(0, 2, 1))  
fitted = model.fit(disp=-1)  
print(fitted.summary())

# Forecast
fc, se, conf = fitted.forecast(11, alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()

'''    
    
    # accuracy__=[]
    # for i in range(input_minimum,input_max+1):
    #     n_input = i
    #     for j in range(output_min,output_max+1):
    #         n_output = j
    #         trainx, trainy, testx, testy = split_data_toTrainAndTest(new_data,n_input, n_output)
    #         accuracy = evaluate_model(trainx,trainy, testx,testy, n_input)
    #         print (i,j, accuracy)
    #         accuracy__.append(accuracy)
    # print (accuracy__)
    # print (max(accuracy__))




