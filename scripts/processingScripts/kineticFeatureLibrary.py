#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 16:44:09 2019

@author: gbonnet
"""
import pandas as pd
import seaborn as sns
import numpy as np
import pickle,math,sys,re
from scipy.optimize import curve_fit
import matplotlib
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.integrate import simps
from scipy.stats import linregress
from sklearn.preprocessing import MinMaxScaler
import itertools
sys.path.insert(0, '../dataProcessing/')
from miscFunctions import sortSINumerically,reindexDataFrame

rsquaredCutoff = 0.7

def createLinRegDataFrame(df,observableList):
    slopedflist = []
    yinterceptdflist = []
    for observable in observableList:
        slopeResults = []
        yinterceptResults = []
        observableDf = df.loc[observable]
        x = df.columns
        for row in range(observableDf.shape[0]):
            timePartitionedY = observableDf.iloc[row,:]
            #timePartitionedY = observableDf.iloc[row,timeStartIndex:timeEndIndex+1]
            result = linregress(x,timePartitionedY.values.ravel())
            #if result[2]**2 >= rsquaredCutoff:
            slopeResults.append(result[0])
            yinterceptResults.append(result[1])
        slopeDf = pd.Series(slopeResults,index=observableDf.index)
        yinterceptDf = pd.Series(yinterceptResults,index=observableDf.index)
        slopedflist.append(slopeDf)
        yinterceptdflist.append(yinterceptDf)
    fullslopedf = pd.concat(slopedflist,axis=1,keys=observableList)
    fullyinterceptdf = pd.concat(slopedflist,axis=1,keys=observableList)
    fullLinRegDf = pd.concat([fullslopedf,fullyinterceptdf],axis=1,keys=['Slopes','YIntercepts'],names=['FeatureType','Observable'])
    return fullLinRegDf

def createSumDataFrame(df,observableList):
    sumdflist = []
    for observable in observableList:
        sumResults = []
        observableDf = df.loc[observable]
        for row in range(observableDf.shape[0]):
            timePartitionedY = observableDf.iloc[row,:]
            #timePartitionedY = observableDf.iloc[row,timeStartIndex:timeEndIndex+1]
            totalSum = timePartitionedY.sum()
            sumResults.append(totalSum)
        sumDf = pd.Series(sumResults,index=observableDf.index)
        sumdflist.append(sumDf)
    sumDf = pd.concat(sumdflist,axis=1,keys=observableList)
    fullSumDf = pd.concat([sumDf],axis=1,keys=['Sums'],names=['FeatureType','Observable'])
    return fullSumDf

def createIntegralFromFitDataFrame(df,observableList,timeStartIndex,timeEndIndex):
    pass

def createFitParameterDataFrame(df,observableList,timeStartIndex,timeEndIndex):
    pass
