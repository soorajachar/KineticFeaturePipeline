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

#Have not actually used this parameter yet (to remove slope/yintercept kinetic features with a low r2) because doing so means
#that we also have to remove the sum kinetic feature for that timepoint/observable. Need to think about how to implement this
#in a different way
rsquaredCutoff = 0.7

#Creates slope and y intercept kinetic feature dataframes
def createLinRegDataFrame(df,observableList):
    slopedflist = []
    yinterceptdflist = []
    #Go through every observable in the timesliced input data dataframe
    for observable in observableList:
        slopeResults = []
        yinterceptResults = []
        observableDf = df.loc[observable]
        #x is always the actual timepoints
        x = df.columns
        #Go through each row in the timesliced-observablesliced df, and grab all the values of the dataframe along that row
        #(input dataframes have time along each row), use these values as y, calculate y = mx+b, store m and b in lists
        for row in range(observableDf.shape[0]):
            timePartitionedY = observableDf.iloc[row,:]
            result = linregress(x,timePartitionedY.values.ravel())
            #if result[2]**2 >= rsquaredCutoff:
            slopeResults.append(result[0])
            yinterceptResults.append(result[1])
        #slopes and y intercepts get made into dataframes and added to separate lists
        slopeDf = pd.Series(slopeResults,index=observableDf.index)
        yinterceptDf = pd.Series(yinterceptResults,index=observableDf.index)
        slopedflist.append(slopeDf)
        yinterceptdflist.append(yinterceptDf)
    #All slope dataframes from each observable are concatenated
    fullslopedf = pd.concat(slopedflist,axis=1,keys=observableList)
    #All yintercept dataframes from each observable are concatenated
    fullyinterceptdf = pd.concat(yinterceptdflist,axis=1,keys=observableList)
    #Slopes and y intercept dataframes are concatenated columnwise, with the "slope" and "yintercept" strings uses to describe their "featuretype"
    fullLinRegDf = pd.concat([fullslopedf,fullyinterceptdf],axis=1,keys=['Slopes','YIntercepts'],names=['FeatureType','Observable'])
    return fullLinRegDf

#Creates sum kinetic feature dataframe
def createSumDataFrame(df,observableList):
    sumdflist = []
    #Go through every observable in the timesliced input data dataframe
    for observable in observableList:
        sumResults = []
        observableDf = df.loc[observable]
        #Go through each row in the timesliced-observablesliced df, and grab all the values of the dataframe along that row
        #(input dataframes have time along each row), calculate the sum of the dataframe along the row (sum of all values in that timepoint region)
        for row in range(observableDf.shape[0]):
            timePartitionedY = observableDf.iloc[row,:]
            totalSum = timePartitionedY.sum()
            sumResults.append(totalSum)
        sumDf = pd.Series(sumResults,index=observableDf.index)
        sumdflist.append(sumDf)
    #All sum dataframes from each observable are concatenated
    sumDf = pd.concat(sumdflist,axis=1,keys=observableList)
    #Sum dataframe is "concatenated" again columnwise, this time to add in the two new levels (featuretype and observable)
    fullSumDf = pd.concat([sumDf],axis=1,keys=['Sums'],names=['FeatureType','Observable'])
    return fullSumDf

def createIntegralFromFitDataFrame(df,observableList,timeStartIndex,timeEndIndex):
    pass

def createFitParameterDataFrame(df,observableList,timeStartIndex,timeEndIndex):
    pass
