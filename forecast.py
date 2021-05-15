#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 16:45:14 2020

@author: vasu
"""

'''
Created on 24-Feb-2020

@author: toto
'''

import pandas as pd
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
#from sklearn.metrics import accuracy_score
#from pandas.core.frame import DataFrame
import statistics as stat
from sklearn.neural_network import MLPRegressor
import scipy
import numpy as np

                

def getEmbedd(data,dim,feature): 
    s=pd.DataFrame()
    for i in range(data.shape[0]-dim):
        s=s.append(pd.DataFrame(data.loc[i:i+dim,feature].values.ravel()).reset_index(drop=True).T, ignore_index=True)
    s = s.sample(frac=1).reset_index(drop=True)
    tgt = (dim+1)*len(feature)-1   
    return (s.loc[:,s.columns!=tgt].values, s.loc[:,s.columns==tgt].values.ravel())

def reduceResolution(data,res=1):
    index = [res *i for i in range(int(data.shape[0]/res))]
    return data.loc[index,:].reset_index(drop=True)
  
def forecast(data,dim,model,features):  
    X,Y = getEmbedd(data,dim,features)
    kf = KFold(n_splits=5)    
    Tr,Ts = [],[]
    for train_index,test_index in kf.split(X,Y):
        model=model.fit(X[train_index],Y[train_index])
        Ts.append(round(model.score(X[test_index],Y[test_index]),2))
        Tr.append(round(model.score(X[train_index],Y[train_index]),2))
    
    print(round(stat.mean(Ts),2),',',round(stat.mean(Tr),2))
       
mat = scipy.io.loadmat('./PWV_2010from_WS_2_withGradient.mat')
data = pd.DataFrame(mat.get("DATA", []),columns = ['doy','hour','minute','temperature','solar_radiation','relative_humidity','rain','dew_point_temp','pwv','Unknown1','Unknown2','Unknown3','Unknown4'])
data['sin_time'] = np.sin(2*np.pi*(data.hour*60+data.minute)/(60.0*24.0))
data['cos_time'] = np.cos(2*np.pi*(data.hour*60+data.minute)/(60.0*24.0))
data['sin_doy'] = np.sin(2*np.pi*data.doy/365.0)
data['cos_doy'] = np.cos(2*np.pi*data.doy/365.0)

data = reduceResolution(data,15)
#print(data.loc[35039],data.shape)

#lm=linear_model.LinearRegression()
#lm=SVR()
#lm=SVR(kernel='linear')
lm = MLPRegressor(max_iter=1000)
#lm = KNeighborsRegressor(n_neighbors=20)

features = ['sin_time','cos_time','solar_radiation']
#features = ['hour','minute','solar_radiation']
#features = ['solar_radiation']
for dim in range(1,10):  forecast(data,dim,lm,features)
    




    
   

