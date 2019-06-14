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

idx = pd.IndexSlice
df = pickle.load(open('fullFeatureDf-20190412-PeptideComparison_OT1_Timeseries_18-raw.pkl','rb'))
test = np.matrix([[0.3,0.2],[0,0],[0.1,0.6],[0.2,0],[0.1,0.3],[0.5,0.3],[0.1,0.1],[0.2,0.2]])
dfWithResponse = df.iloc[[0,1,2,3,4,5,6,8,9,10,11,12],:]
peptides = list(dfWithResponse.index.get_level_values('Peptide'))
uniquePeptides = pd.unique(peptides)
pairwisePeptides = [*combinations(uniquePeptides,2)]
kineticFeatures = df.loc[:,idx['cyt','NotApplicable','CytokineConcentration',:,:,'IndividualObservation',:]]
x_grid = np.linspace(0, 1, 100)
distanceMetrics = [dist.euclidean,dist.cityblock,dist.chebyshev]
distanceMetricNames = ['Euclidean','Manhattan','Chebyshev']
for column in range(kineticFeatures.shape[1]):
    kineticFeature = kineticFeatures.iloc[:,column]
    for distanceMetric,distanceName in zip(distanceMetrics,distanceMetricNames):
        totaldistance = 0
        for pairwisePeptide in pairwisePeptides:
            peptide1 = kineticFeature.loc[pairwisePeptide[0]]
            peptide2 = kineticFeature.loc[pairwisePeptide[1]]
            try:
                kde1 = gaussian_kde(peptide1)
                kde2 = gaussian_kde(peptide2)
                pdf1 = kde1.evaluate(x_grid)
                pdf2 = kde2.evaluate(x_grid)
                distance = distanceMetric(pdf1,pdf2)
            except:
                distance = 0
            else:
                kde1 = gaussian_kde(peptide1)
                kde2 = gaussian_kde(peptide2)
                pdf1 = kde1.evaluate(x_grid)
                pdf2 = kde2.evaluate(x_grid)
                distance = distanceMetric(pdf1,pdf2)
            totaldistance += distance
        avgdistance = totaldistance / len(pairwisePeptides)
