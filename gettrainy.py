# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 17:14:13 2020

@author: Administrator
"""

#%% For extracting trainY (dbl,bs,co,su,vd) from files.

import os
import pandas as pd
import numpy

def gettrainy():
    rootDir = 'vent map_derivation_anonymized/gold_stnd_files'
    
    Trainy=pd.DataFrame()
    i=0
    for dirName, subdirList, fileList in os.walk(rootDir):
        
        for fname in fileList:
            dirName=dirName.replace('\\','/')
            data = dirName+'/'+fname
            dd=pd.read_csv(data)
            Targetdata=dd.loc[:,['dbl','bs','co','su','vd']]
            Trainy = [Trainy, Targetdata]
            Trainy = pd.concat(Trainy)
            i +=1

    return Trainy

trainy=gettrainy().to_numpy()
numpy.save('trainy_for_GAN',trainy)

