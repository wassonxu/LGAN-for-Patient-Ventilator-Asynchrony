# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 21:32:34 2020

@author: Administrator
"""
 

#%% For extracting metadata (I-Time, e-time,TVi, TVe,ipAUC, epAUC) from files.

# By change line 68 we can select the right features for our data
import os
import pandas as pd
from ventmap.breath_meta import get_file_breath_meta
import numpy
def getBN():
    '''

    Returns
    -------
    BN : Numpy array
        DESCRIPTION: give the gold standard data index

    '''
    
    rootDir = 'D:/DL/11AAnewest/vent map_derivation_anonymized/gold_stnd_files'
    breathdata = pd.DataFrame()
    BN = []
    for dirName, subdirList, fileList in os.walk(rootDir):
        
        # print('Found directory: %s' % dirName)
    
        for fname in fileList:
            dirName=dirName.replace('\\','/')
            data = dirName+'/'+fname
            dd=pd.read_csv(data)
            Targetd=dd.loc[:,['BN']]
            Targetd = pd.DataFrame.to_numpy(Targetd)
            
            BN.append(Targetd)
    BN=numpy.array(BN)
    return BN

BN = getBN()
data = BN
dd = list()
for i in data:
    dd.append(i[0])
    dd.append(i[-1])
dd = numpy.array(dd)
dd=dd.reshape(37,2)
   

#%% Here we get the flow and pressure
'''
from io import open
import numpy as np
from ventmap.raw_utils import process_breath_file, read_processed_file
import pandas as pd
import numpy
# This function will output 2 files. The first will just contain raw breath data
# the other will contain higher level processed data. In order to re-load the
# saved data we just need to specify the path to the raw file




# pressure_=numpy.append(pressure_one,flow_one,axis = 0)
# pressure_one.append([1,2,2,2,2,4])
# Traindata = [pressure_one, flow_one]
# Traindata = pd.concat(Traindata)

def getRawdata():
    rootDir = 'D:/DL/11AAnewest/vent map_derivation_anonymized/cohort_derivation'
    count = 0
    pressure_one, flow_one = np.empty((1,800)), np.empty((1,800))
    for dirName, subdirList, fileList in os.walk(rootDir):
        
        # print('Found directory: %s' % dirName)
    
        for fname in fileList:
            dirName=dirName.replace('\\','/')
            data = dirName+'/'+fname
            
            process_breath_file(open(data), False, 'new_filename')           
            raw_filepath_name = 'new_filename.raw.npy'
            for breath in read_processed_file("new_filename.raw.npy","new_filename.processed.npy"):
                # breath data is output in dictionary format
                
                real_bn, flow, pressure =breath['rel_bn'], breath['flow'], breath['pressure']
                if real_bn >= dd[count][0] and real_bn<= dd[count][1]:
                    len_p = len(pressure)
                    N = 800 - len_p
                    pressure = np.pad(pressure,(0,N),'constant')
                    pressure_one = numpy.append(pressure_one, np.array(pressure).reshape(1,len(pressure)),axis = 0)
                    
                    flow = np.pad(flow,(0,N),'constant')
                    flow_one = numpy.append(flow_one,np.array(flow).reshape(1,len(flow)),axis = 0)
            

            count = count + 1
            print ('count is',count)
            
    # Traindata = Traindata.to_numpy()
    return pressure_one, flow_one 
pressure, flow = getRawdata()

length = len(pressure)
leng_flow = len(flow)
numpy.save('pressure',pressure[1:])
numpy.save('flow',flow[1:])
'''
#%% Here we extracted the all of the features
'''
def getTraindata():
    rootDir = 'D:/DL/11AAnewest/vent map_derivation_anonymized/cohort_derivation'
    count = 0
    Traindata=pd.DataFrame()
    for dirName, subdirList, fileList in os.walk(rootDir):
        
        # print('Found directory: %s' % dirName)
    
        for fname in fileList:
            dirName=dirName.replace('\\','/')
            data = dirName+'/'+fname
            breath_meta = get_file_breath_meta(data, to_data_frame=True)
            breath_meta = breath_meta.loc[(breath_meta['BN'] >=dd[count][0]) & (breath_meta['BN'] <= dd[count][1])]
            # Extract the wanted feature, such as 'iTime','eTime','tvi','tve'
            # breath_meta=breath_meta.loc[:, ['iTime','eTime','tvi','tve','ipAUC','epAUC']]
            breath_meta=breath_meta.loc[:, :]
            Traindata = [Traindata, breath_meta]
            Traindata = pd.concat(Traindata)
            count = count + 1
            print ('count is',count)
            
    # Traindata = Traindata.to_numpy()
    Traindata = Traindata
    return Traindata

Traindata = getTraindata()
from pandas import DataFrame as df
Traindata.to_csv('train_all.csv')
df = pd.read_csv('train_all.csv')
# numpy.save('trainx_all',Traindata)
'''        
#%% Here we extracted the all of the features and the retrospective
'''
retro = 1

def getTraindata():
    rootDir = 'D:/DL/11AAnewest/vent map_derivation_anonymized/cohort_derivation'
    count = 0
    Traindata=pd.DataFrame()
    for dirName, subdirList, fileList in os.walk(rootDir):
        
        # print('Found directory: %s' % dirName)
    
        for fname in fileList:
            dirName=dirName.replace('\\','/')
            data = dirName+'/'+fname
            print ('processing on file:',fname)
            breath_meta = get_file_breath_meta(data, to_data_frame=True)
            breath_meta = breath_meta.loc[(breath_meta['BN'] >=dd[count][0]) & (breath_meta['BN'] <= dd[count][1])]
            # Extract the wanted feature, such as 'iTime','eTime','tvi','tve'
            # breath_meta=breath_meta.loc[:, ['iTime','eTime','tvi','tve','ipAUC','epAUC']]
            # for i in breath_meta.index:
            #     breath_meta.loc[i]
            # breath_meta=breath_meta.loc[:, :]
            columns_name= []
            for col in breath_meta.columns:
                columns_name.append ('-1_'+col)
            
            # arr = numpy.zeros(breath_meta.shape)
            arr1 = breath_meta.values
            print ('The data shape of this file is',breath_meta.values.shape)
            arr_insert = numpy.zeros((1,arr1.shape[1]))
            arr2 = numpy.concatenate((arr_insert,arr1),axis =0)

            arr2 = arr2[:-1]
            breath_meta_retro_1 = pd.DataFrame(arr2, columns = columns_name) 
            
            #reset index 
            breath_meta_retro_1 = breath_meta_retro_1.reset_index(drop=True)
            breath_meta = breath_meta.reset_index(drop=True)
            breath_meta_retro = breath_meta.join(breath_meta_retro_1)
            
            
            
            
            Traindata = [Traindata, breath_meta_retro]
            Traindata = pd.concat(Traindata)
            count = count + 1
            print ('count is',count)
            
    # Traindata = Traindata.to_numpy()
    Traindata = Traindata
    return Traindata

Traindata = getTraindata()
from pandas import DataFrame as df
Traindata.to_csv('train_all_retro_1.csv')
df = pd.read_csv('train_all_retro_1.csv')
# numpy.save('trainx_all',Traindata)
'''
#%% Here we extracted the all of the features and the retrospective
'''
retro = 2

def getTraindata():
    rootDir = 'D:/DL/11AAnewest/vent map_derivation_anonymized/cohort_derivation'
    count = 0
    Traindata=pd.DataFrame()
    for dirName, subdirList, fileList in os.walk(rootDir):
        
        # print('Found directory: %s' % dirName)
    
        for fname in fileList:
            dirName=dirName.replace('\\','/')
            data = dirName+'/'+fname
            print ('processing on file:',fname)
            breath_meta = get_file_breath_meta(data, to_data_frame=True)
            breath_meta = breath_meta.loc[(breath_meta['BN'] >=dd[count][0]) & (breath_meta['BN'] <= dd[count][1])]
            # Extract the wanted feature, such as 'iTime','eTime','tvi','tve'
            # breath_meta=breath_meta.loc[:, ['iTime','eTime','tvi','tve','ipAUC','epAUC']]
            # for i in breath_meta.index:
            #     breath_meta.loc[i]
            # breath_meta=breath_meta.loc[:, :]
            columns_name= []
            for col in breath_meta.columns:
                columns_name.append ('-1_'+col)
            
            # arr = numpy.zeros(breath_meta.shape)
            arr1 = breath_meta.values
            print ('The data shape of this file is',breath_meta.values.shape)
            arr_insert = numpy.zeros((1,arr1.shape[1]))
            arr2 = numpy.concatenate((arr_insert,arr1),axis =0)

            arr2 = arr2[:-1]
            breath_meta_retro_1 = pd.DataFrame(arr2, columns = columns_name) 
            
            #reset index 
            breath_meta_retro_1 = breath_meta_retro_1.reset_index(drop=True)
            breath_meta = breath_meta.reset_index(drop=True)
            breath_meta_retro = breath_meta.join(breath_meta_retro_1)
            
            # retro ==2 
            columns_name= []
            for col in breath_meta.columns:
                columns_name.append ('-2_'+col)
            arr_retro_2 = breath_meta.values
            print ('The data shape of this file is',breath_meta.values.shape)
            arr_insert = numpy.zeros((2,arr1.shape[1]))
            arr_retro_2_right = numpy.concatenate((arr_insert,arr_retro_2),axis =0)
            
            arr_retro_2_right = arr_retro_2_right[:-2]
            breath_meta_retro_2 = pd.DataFrame(arr_retro_2_right, columns = columns_name) 
            
            breath_meta_retro_1 = breath_meta_retro.reset_index(drop=True)
            breath_meta_retro_2 = breath_meta_retro_2.reset_index(drop=True)
            breath_meta_retro_2_ = breath_meta_retro_1.join(breath_meta_retro_2)
            
            
            
            Traindata = [Traindata, breath_meta_retro_2_]
            Traindata = pd.concat(Traindata)
            count = count + 1
            print ('count is',count)
            
    # Traindata = Traindata.to_numpy()
    Traindata = Traindata
    return Traindata

Traindata = getTraindata()
from pandas import DataFrame as df
Traindata.to_csv('train_all_retro_2.csv')
df = pd.read_csv('train_all_retro_2.csv')
# numpy.save('trainx_all',Traindata)
'''

#%% Here we extracted the all of the features and the retrospective

retro = 3

def getTraindata():
    rootDir = 'D:/DL/11AAnewest/vent map_derivation_anonymized/cohort_derivation'
    count = 0
    Traindata=pd.DataFrame()
    for dirName, subdirList, fileList in os.walk(rootDir):
        
        # print('Found directory: %s' % dirName)
    
        for fname in fileList:
            dirName=dirName.replace('\\','/')
            data = dirName+'/'+fname
            print ('processing on file:',fname)
            breath_meta = get_file_breath_meta(data, to_data_frame=True)
            breath_meta = breath_meta.loc[(breath_meta['BN'] >=dd[count][0]) & (breath_meta['BN'] <= dd[count][1])]
            # Extract the wanted feature, such as 'iTime','eTime','tvi','tve'
            # breath_meta=breath_meta.loc[:, ['iTime','eTime','tvi','tve','ipAUC','epAUC']]
            # for i in breath_meta.index:
            #     breath_meta.loc[i]
            # breath_meta=breath_meta.loc[:, :]
            columns_name= []
            for col in breath_meta.columns:
                columns_name.append ('-1_'+col)
            
            # arr = numpy.zeros(breath_meta.shape)
            arr1 = breath_meta.values
            print ('The data shape of this file is',breath_meta.values.shape)
            arr_insert = numpy.zeros((1,arr1.shape[1]))
            arr2 = numpy.concatenate((arr_insert,arr1),axis =0)

            arr2 = arr2[:-1]
            breath_meta_retro_1 = pd.DataFrame(arr2, columns = columns_name) 
            
            #reset index 
            breath_meta_retro_1 = breath_meta_retro_1.reset_index(drop=True)
            breath_meta = breath_meta.reset_index(drop=True)
            breath_meta_retro = breath_meta.join(breath_meta_retro_1)
            
            # retro ==2 
            columns_name= []
            for col in breath_meta.columns:
                columns_name.append ('-2_'+col)
            arr_retro_2 = breath_meta.values
            print ('The data shape of this file is',breath_meta.values.shape)
            arr_insert = numpy.zeros((2,arr1.shape[1]))
            arr_retro_2_right = numpy.concatenate((arr_insert,arr_retro_2),axis =0)
            
            arr_retro_2_right = arr_retro_2_right[:-2]
            breath_meta_retro_2 = pd.DataFrame(arr_retro_2_right, columns = columns_name) 
            
            breath_meta_retro_1 = breath_meta_retro.reset_index(drop=True)
            breath_meta_retro_2 = breath_meta_retro_2.reset_index(drop=True)
            breath_meta_retro_2_ = breath_meta_retro_1.join(breath_meta_retro_2)
            
            
            # retro ==3
            columns_name= []
            for col in breath_meta.columns:
                columns_name.append ('-3_'+col)
            arr_retro_3 = breath_meta.values
            print ('The data shape of this file is',breath_meta.values.shape)
            arr_insert = numpy.zeros((3,arr1.shape[1]))
            arr_retro_3_right = numpy.concatenate((arr_insert,arr_retro_3),axis =0)
            
            arr_retro_3_right = arr_retro_3_right[:-3]
            breath_meta_retro_3 = pd.DataFrame(arr_retro_3_right, columns = columns_name) 
            
            breath_meta_retro_3_0 = breath_meta_retro_2_.reset_index(drop=True)
            breath_meta_retro_3_1 = breath_meta_retro_3.reset_index(drop=True)
            breath_meta_retro_3_ = breath_meta_retro_3_0.join(breath_meta_retro_3_1)
            
            
            
            Traindata = [Traindata, breath_meta_retro_3_]
            Traindata = pd.concat(Traindata)
            count = count + 1
            print ('count is',count)
            
    # Traindata = Traindata.to_numpy()
    Traindata = Traindata
    return Traindata

Traindata = getTraindata()
from pandas import DataFrame as df
Traindata.to_csv('train_all_retro_3.csv')
df = pd.read_csv('train_all_retro_3.csv')
# numpy.save('trainx_all',Traindata)

#%% Here we seperate the BSA, DBL, ARTifact, and NORMAL breath
'''
import numpy as np
import pandas as pd

trainX = pd.read_csv('train_all.csv')
trainY = np.load('trainy_all.npy')

def Change_array_to_list(label_data):
    list_data = []
    for i in label_data:
        list_data.append(str(i))
    return list_data

def check_AB(label_data):
    # check is there any abnormal data
    Temp = []
    for i in label_data:
        Temp.append(str(i))
    unique, counts = np.unique (Temp, return_counts=True)
    return unique, counts

# check is there any abnormal data
unique ,counts = check_AB(trainY)

#try to find the index of the abnormal data
List_trainY = Change_array_to_list(trainY)
indices = [i for i, x in enumerate(List_trainY) if x == "[1 1 0 0 0]"] # return the bad value indices = 99,118,186,2312,2314, 2395

# then I am gonna delete the bad value in trainY and trainX
# trainyyy.remove('[1 1 0 0 0]') # remove from trainy  for 6 times
# trainy=np.array([np.array(xi) for xi in trainyyy])

#delete the abnormal data
All_normal_trainY=np.delete(trainY,indices, axis=0)
All_normal_trainX = trainX.drop(indices)


All_normal_trainX.to_csv('Seperated_data\All_normal_trainX')
np.save('Seperated_data\All_normal_trainY',All_normal_trainY)


List_trainY = Change_array_to_list(All_normal_trainY)
indices_DTA = [i for i, x in enumerate(List_trainY) if x == "[1 0 0 0 0]"]
indices_BSA = [i for i, x in enumerate(List_trainY) if x == "[0 1 0 0 0]"]
indices_Normal = [i for i, x in enumerate(List_trainY) if x == "[0 0 0 0 0]"]
indices_ARTifact = [i for i, x in enumerate(List_trainY) if x == "[0 0 1 0 0]" or x== "[0 0 0 1 0]" or x == "[0 0 0 0 1]"]

All_Normal_trainY = All_normal_trainY[indices_Normal]
All_Normal_trainX = All_normal_trainX.iloc[indices_Normal] 

All_BSA_trainY = All_normal_trainY[indices_BSA]
All_BSA_trainX = All_normal_trainX.iloc[indices_BSA] 

All_DTA_trainY = All_normal_trainY[indices_DTA]
All_DTA_trainX = All_normal_trainX.iloc[indices_DTA]

All_ARTifact_trainY = All_normal_trainY[indices_ARTifact]
All_ARTifact_trainX = All_normal_trainX.iloc[indices_ARTifact]


All_Normal_trainX.to_csv('Seperated_data\All_Normal_trainX')
All_BSA_trainX.to_csv('Seperated_data\All_BSA_trainX')
All_DTA_trainX.to_csv('Seperated_data\All_DTA_trainX')
All_ARTifact_trainX.to_csv('Seperated_data\All_ARTifact_trainX')

np.save('Seperated_data\All_Normal_trainY',All_Normal_trainY)
np.save('Seperated_data\All_BSA_trainY',All_BSA_trainY)
np.save('Seperated_data\All_DTA_trainY',All_DTA_trainY)
np.save('Seperated_data\All_ARTifact_trainY',All_ARTifact_trainY)
'''

#%% Example of walking through the files

# # Import the os module, for the os.walk function
# import os
# import pandas as pd
# # Set the directory you want to start from
# rootDir = 'vent map_derivation_anonymized/cohort_derivation/gold_stnd_files'
# # aa = os.walk(rootDir)
# Traindata=pd.DataFrame()
# Targetdata=pd.DataFrame()


# for dirName, subdirList, fileList in os.walk(rootDir):
    
#     # print('Found directory: %s' % dirName)

#     for fname in fileList:
#         dirName=dirName.replace('\\','/')
#         data = dirName+'/'+fname
#         dd=pd.read_csv(data)
#         Traind=dd.iloc[:,4:8]
#         Traindata = [Traindata, Traind]
#         Traindata = pd.concat(Traindata)
        
#         Targetd=dd.loc[:,['dbl','bs','co','su','vd']]

#         print (fname)
#         Targetdata = [Targetdata, Targetd]
#         Targetdata = pd.concat(Targetdata)
        
        
# '''       
#         aa
#         dd=pandas.read_csv("dirName/fname" )
#         Traindata=dd.iloc[:,4:8]
#         Targetdata=dd.loc[:,['dbl','bs','co','su','vd']]
        
#         # print('\t%s' % fname)
#         '''