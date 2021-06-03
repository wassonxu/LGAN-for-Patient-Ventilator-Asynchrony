# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 18:26:39 2020

@author: Administrator
"""

#%% check the data and found that there are dbl and bsa exist at the same time!
import numpy as np
import pandas as pd
trainyy = np.load('trainy_all.npy')  # testy.npy


# ## transfer to str type to count the same value
# trainyyy= []
# for i in trainyy:
#     trainyyy.append(str(i))
#     # trainyyy[i]= str(j)
    
# unique,counts = np.unique(trainyyy,return_counts = True)  # count the number of different class 
    
# # there are dbl and bsa exist at the same time! it cannot be possible because it is a classification problem




# #try to find the index of the abnormal data
# indices = [i for i, x in enumerate(trainyyy) if x == "[1 1 0 0 0]"] # return the bad value  indices = 99,118,186,2312,2314, 2395


# # then I am gonna delete the bad value in trainy and trainx
# # trainyyy.remove('[1 1 0 0 0]') # remove from trainy  for 6 times

# # trainy=np.array([np.array(xi) for xi in trainyyy])

# #delete the abnormal data
# t=np.delete(trainyy,indices, axis=0)
# np.save('trainy_after_Check',t)

# # for retro1
# trainx = pd.read_csv('train_all_retro_1.csv')  # testy.npy
# New_testx=trainx.drop(indices)
# New_testx.to_csv('train_all_retro_1_after_check.csv')

# # for retro2
# trainx = pd.read_csv('train_all_retro_2.csv')  # testy.npy
# New_testx=trainx.drop(indices)
# New_testx.to_csv('train_all_retro_2_after_check.csv')

# # for retro3
# trainx = pd.read_csv('train_all_retro_3.csv')  # testy.npy
# New_testx=trainx.drop(indices)
# New_testx.to_csv('train_all_retro_3_after_check.csv')

df = pd.read_csv("train_all_retro_2_after_check.csv")
