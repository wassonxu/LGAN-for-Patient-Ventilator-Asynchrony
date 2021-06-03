# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 08:44:58 2021

@author: Administrator
"""

import numpy as np
from utilities.utilities import Change_array_to_list
from collections import Counter

flow = np.load('flow.npy')
pressure = np.load('pressure.npy')
Trainy = np.load ('trainy_all_for_flow_and_pressure.npy')

print (flow.shape)
print (Trainy)



trainy= []
for i in Trainy:
    trainy.append(str(i))
    # trainyyy[i]= str(j)
    
unique,counts = np.unique(trainy,return_counts = True)  # count the number of different class 


#try to find the index of the abnormal data
indices = [i for i, x in enumerate(trainy) if x == "[1 1 0 0 0]"] # return the bad value  indices


t=np.delete(trainy,indices, axis=0)
np.save('Seperated_data_flow_and_pressure/trainy',t)

New_flow=np.delete(flow,indices, axis=0)
New_pressure = np.delete (pressure, indices, axis =0)




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


trainy_BSA = get_BSA_and_normal(t)

trainy_DTA = get_DTA_and_normal(t)

arr = Change_array_to_list(trainy_BSA)
unique,counts = np.unique(arr,return_counts = True)

arrB= Change_array_to_list(trainy_DTA)
unique,counts = np.unique(arrB,return_counts = True)


np.save('Seperated_data_flow_and_pressure/trainy_BSA',arr)
np.save('Seperated_data_flow_and_pressure/trainy_DTA',arrB)