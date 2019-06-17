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
folderName = '20190412-PeptideComparison_OT1_Timeseries_18'
qualitySeparationMetricName = 'mutualInfo'
qualityMetricDf = pickle.load(open('qualitySeparationMetric-'+folderName+'-'+qualitySeparationMetricName+'.pkl','rb')).loc[idx[dataTypeList]]
featureDf = pickle.load(open('fullFeatureDf-'+folderName+'-preprocessed-qualityOrdered.pkl','rb'))
numberOfTopScoring = 10

sortedQualityMetricDf = qualityMetricDf.sort_values('Quality Separation',ascending=False)
for cytokine in ['IL-2','IL-6','TNFa','IFNg','IL-17A']:
    plottingSubsetQuality = sortedQualityMetricDf.loc[idx['cyt',:,:,:,:,'IndividualObservation',cytokine],:]
    plottingColumnsQuality,plottingDfListQuality = grabTopScoringFeatureData(plottingSubsetQuality,featureDf)
    plottingDfQuality = pd.DataFrame(np.matrix(plottingDfListQuality).T,index=featureDf.index,columns=plottingColumnsQuality)
    plottingDfQuality.columns.name = 'Feature'
    plottingDf = plottingDfQuality.stack().to_frame('Feature Value').reset_index()
    fg = sns.FacetGrid(plottingDf,sharey=False,legend_out=True,col='Feature',hue='Peptide',col_wrap=10)
    fg.map(sns.kdeplot,'Feature Value',shade=True)
    plt.savefig('temp/individualComparisonCytokine'+cytokine+'.png')
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
