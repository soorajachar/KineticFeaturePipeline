#!/usr/bin/env python3 
import json,pickle,math,matplotlib,sys,os,string
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance as dist
from scipy.stats import gaussian_kde
from itertools import combinations

idx = pd.IndexSlice

def grabTopScoringFeatureData(plottingSubset,featureDf):
    mi_df = []
    tupleList = []
    plottingDfList = []

    for row in range(plottingSubset.shape[0]):
        currentrow = plottingSubset.iloc[row,:].name
        featureDf2 = featureDf.copy()
        for indexer in currentrow:
            featureDf2 = featureDf2.loc[:,indexer]
        plottingDfList.append(featureDf2)
        tupleList.append(currentrow)
    plottingColumnsBeforeModification = pd.MultiIndex.from_tuples(tupleList,names=featureDf.columns.names)
    plottingColumns = []
    for individualTuple in tupleList:
        newTuple = []
        for element,j in zip(individualTuple,range(len(individualTuple))):
            if '/' in str(element):
                populationDivisionIndices = [i for i,c in enumerate(element) if c=='/']
                temp = element[populationDivisionIndices[-2]+1:]
                newTuple.append(temp)
            else:
                newTuple.append(str(element))
        plottingColumns.append('-\n'.join(newTuple))
    return plottingColumns,plottingDfList

def grabTopScoringFeaturePlottingData(plottingSubset,featureDf):
    plottingList = []

    for row in range(plottingSubset.shape[0]):
        currentrow = plottingSubset.iloc[row,:].name
        featureDf2 = featureDf.copy()
        for indexer in currentrow:
            featureDf2 = featureDf2.loc[:,indexer]
        if 'Slopes' in currentrow:
            plottingList.append(currentrow)
            yintlist = []
            for val in currentrow:
                if val == 'Slopes':
                    yintlist.append('YIntercepts')
                else:
                    yintlist.append(val)
            plottingList.append(tuple(yintlist))
    
    return plottingList

def abline(slope, intercept,timeSliceStart,timeSliceEnd,qualityColor,quantityStyle):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array([timeSliceStart,timeSliceEnd])
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, color=qualityColor,linestyle=quantityStyle)

folderName = '20190412-PeptideComparison_OT1_Timeseries_18'
qualitySeparationMetricName = 'mutualInformation'
dataTypeList = slice(None)
qualityMetricDf = pickle.load(open('qualitySeparationMetric-'+folderName+'-'+qualitySeparationMetricName+'.pkl','rb')).loc[idx[dataTypeList]]
featureDf = pickle.load(open('fullFeatureDf-'+folderName+'-preprocessed-qualityOrdered.pkl','rb'))
rawFeatureDf = pickle.load(open('fullFeatureDf-'+folderName+'-raw.pkl','rb'))
rawDataDf = np.log10(pickle.load(open('../kineticFeatureInput/cytokineConcentrationPickleFile-20190412-PeptideComparison_OT1_Timeseries_18-modified.pkl','rb'))).loc['IL-6']
numberOfTopScoring = 10
"""
sortedQualityMetricDf = qualityMetricDf.sort_values('Quality Separation',ascending=False)
plottingSubsetQuality = sortedQualityMetricDf.iloc[:numberOfTopScoring,:]
tupleList = grabTopScoringFeaturePlottingData(plottingSubsetQuality,rawFeatureDf)
top10RawFeatureDfQuality = rawFeatureDf.loc[:,idx[tupleList]]
linestyle_tuple = [
     ('solid',                 (0, ())),
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),

     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]
linecolors = ['k','c','y','r','g','b','m']
lineStyleDict = {}
lineColorDict = {}
qualities = list(pd.unique(top10RawFeatureDfQuality.index.get_level_values('Peptide')))
quantities = list(pd.unique(top10RawFeatureDfQuality.index.get_level_values('Concentration')))
for qual,i in zip(qualities,range(len(qualities))):
    lineColorDict[qual] = linecolors[i]
for quan,i in zip(quantities,range(len(quantities))):
    lineStyleDict[quan] = linestyle_tuple[i][1]
fig = plt.figure()
for row in range(top10RawFeatureDfQuality.shape[0]):
    name = top10RawFeatureDfQuality.iloc[row,:].name
    columnname = top10RawFeatureDfQuality.iloc[:,0].name
    timeSliceStart = columnname[-4]
    timeSliceEnd = columnname[-3]
    plt.plot(rawDataDf.columns,rawDataDf.iloc[row,:])
    print(top10RawFeatureDfQuality.iloc[row,1])
    abline(top10RawFeatureDfQuality.iloc[row,0],top10RawFeatureDfQuality.iloc[row,1],timeSliceStart,timeSliceEnd,lineColorDict[name[0]],lineStyleDict[name[1]])
plt.show()
"""
dataType = 'cyt'
sortedQualityMetricDf = qualityMetricDf.sort_values('Quality Separation',ascending=False)
plottingSubsetQuality1 = sortedQualityMetricDf.loc[idx[dataType,:,:,:,:,'IndividualObservation',:],:]
plottingSubsetQuality2 = sortedQualityMetricDf.iloc[:numberOfTopScoring,:]
plottingSubsetQuality = pd.concat([plottingSubsetQuality1,plottingSubsetQuality2],axis=0)
#plottingColumnsQuality,plottingDfListQuality = grabTopScoringFeatureData(plottingSubsetQuality,featureDf)
#plottingDfQuality = pd.DataFrame(np.matrix(plottingDfListQuality).T,index=featureDf.index,columns=plottingColumnsQuality)
#plottingDfQuality.columns.name = 'Feature'
#plottingDf = plottingSubsetQuality.stack().to_frame('Mutual Info').reset_index()
plottingDf = plottingSubsetQuality.reset_index()
g = sns.relplot(data=plottingDf,style='Observable',y='Quality Separation',col='DataType',hue='FeatureType',x='TimeSliceEnd',kind='scatter')
#g = sns.relplot(data=plottingDf,style='FeatureType',y='Quality Separation',col='DataType',hue='Observable',x='TimeSliceEnd',kind='scatter')
plt.savefig('temp/individualComparisonCytokine-scatter-'+dataType+'.png')
#plt.show()
"""
sortedQualityMetricDf = qualityMetricDf.sort_values('Quality Separation',ascending=False)
for cytokine in ['IL-2','IL-6','TNFa','IFNg']:
    plottingSubsetQuality = sortedQualityMetricDf.loc[idx['cyt',:,:,:,:,'IndividualObservation',cytokine],:]
    plottingColumnsQuality,plottingDfListQuality = grabTopScoringFeatureData(plottingSubsetQuality,featureDf)
    plottingDfQuality = pd.DataFrame(np.matrix(plottingDfListQuality).T,index=featureDf.index,columns=plottingColumnsQuality)
    plottingDfQuality.columns.name = 'Feature'
    plottingDf = plottingDfQuality.stack().to_frame('Feature Value').reset_index()
    fg = sns.FacetGrid(plottingDf,sharey=False,legend_out=True,col='Feature',hue='Peptide',col_wrap=10)
    fg.map(sns.kdeplot,'Feature Value',shade=True)
    plt.savefig('temp/individualComparisonCytokine'+cytokine+'.png')
    print(cytokine)
    plt.clf()

cytokine = 'top10'
plottingSubsetQuality = sortedQualityMetricDf.iloc[:numberOfTopScoring,:]
plottingColumnsQuality,plottingDfListQuality = grabTopScoringFeatureData(plottingSubsetQuality,featureDf)
plottingDfQuality = pd.DataFrame(np.matrix(plottingDfListQuality).T,index=featureDf.index,columns=plottingColumnsQuality)
plottingDfQuality.columns.name = 'Feature'
plottingDf = plottingDfQuality.stack().to_frame('Feature Value').reset_index()
fg = sns.FacetGrid(plottingDf,sharey=False,legend_out=True,col='Feature',hue='Peptide',col_wrap=int(numberOfTopScoring/2))
fg.map(sns.kdeplot,'Feature Value',shade=True)
plt.savefig('temp/individualComparisonCytokine'+cytokine+'.png')
"""
