#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 16:44:09 2019

@author: gbonnet
"""
import pandas as pd
import seaborn as sns
import numpy as np
import os
import pickle,math,sys,re
from scipy.optimize import curve_fit
import matplotlib
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.integrate import simps
from scipy.stats import linregress
import itertools
from sklearn.feature_selection import mutual_info_classif
sys.path.insert(0, '../dataProcessing/')
from miscFunctions import sortSINumerically,reindexDataFrame
from scipy.spatial import distance as dist
from scipy.stats import gaussian_kde
from itertools import combinations

idx = pd.IndexSlice

#Different distance metrics can go into this dictionary (anything from scipy.spatial.dist should work)
distanceMetricDictionary = {'euclidean':dist.euclidean,'manhattan':dist.cityblock,'chebyshev':dist.chebyshev}

#Quality deconvolution measurement method. "Eithersorted" boolean allows for separate dataframes to be saved for features that order either quality/quantity correctly (True) or just quality (False)
def qualityMetric(standardizedFeatureDf,expNum,folderName,qualitySeparationMetricName,eitherSorted):
    #Using nearest neighbor mutual information metric estimate described here: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0087357#pone.0087357-Abramowitz1 
    #Default k value in sklearn package is 3 (3rd nearest neighbor); if fewer than 4 concentrations per peptide it should be adjusted to n-1 where n is the number of concentrations per peptide
    #For some reason, the most independent distributions have the highest mutual information. Needs to be looked into
    #Also does not handle multiple measurements with the same values well
    if qualitySeparationMetricName == 'mutualInformation':
        #Grab all peptides in dataframe, use as classes for mutual information method
        peptides  = list(standardizedFeatureDf.index.get_level_values('Peptide'))
        newIndex = pd.MultiIndex.from_tuples(list(standardizedFeatureDf.columns),names=standardizedFeatureDf.columns.names)
        qualitySeparationDf = pd.Series(mutual_info_classif(standardizedFeatureDf.values,peptides),index=newIndex)
    #Tried using distance metrics to determine quality deconvolution, but they did not work very well. Needs to be revisited
    else:
        distanceMetric = distanceMetricDictionary[qualitySeparationMetricName]
        peptides  = list(standardizedFeatureDf.index.get_level_values('Peptide'))
        uniquePeptides = pd.unique(peptides)
        pairwisePeptides = [*combinations(uniquePeptides,2)]
        newIndex = pd.MultiIndex.from_tuples(list(standardizedFeatureDf.columns),names=standardizedFeatureDf.columns.names)
        x_grid = np.linspace(0, 1, 100)
        avgdistancelist = []
        for column in range(standardizedFeatureDf.shape[1]):
            kineticFeature = standardizedFeatureDf.iloc[:,column]
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
            avgdistancelist.append(avgdistance)
            print(str(column)+'/'+str(standardizedFeatureDf.shape[1])+' is done')
        qualitySeparationDf = pd.Series(np.array(avgdistancelist).T,index = newIndex)
    
    #Higher ->better separated
    qualitySeparationDf = qualitySeparationDf.to_frame('Quality Separation')
    if eitherSorted:
        with open('kineticFeatureOutput/qualitySeparationMetric-'+folderName+'-'+qualitySeparationMetricName+'-either.pkl','wb') as f:
            pickle.dump(qualitySeparationDf,f)
    else:
        with open('kineticFeatureOutput/qualitySeparationMetric-'+folderName+'-'+qualitySeparationMetricName+'.pkl','wb') as f:
            pickle.dump(qualitySeparationDf,f)

#Options for quantity separation metrics are limited because we are no longer dealing with distances between distributions (so we cannot use scipy.distances) but rather distances between individual points
#Using Average CVs of peptide concentrations (would be nice to get a more robust measure resistant to outliers than the average; something besides the median)
def quantityMetric(standardizedFeatureDf,expNum,folderName,quantitySeparationMetricName,eitherSorted):
    if quantitySeparationMetricName == 'CV':
        classifications  = list(standardizedFeatureDf.index.get_level_values('Peptide'))
        print(standardizedFeatureDf)
        currentCV = standardizedFeatureDf.groupby(['Peptide']).std()/abs(standardizedFeatureDf.groupby(['Peptide']).mean())
        quantitySeparationDf = currentCV.mean()
    #Higher -> better separated
    quantitySeparationDf = quantitySeparationDf.to_frame('Quantity Separation')
    print(quantitySeparationDf)
    if eitherSorted:
        with open('kineticFeatureOutput/quantitySeparationMetric-'+folderName+'-'+quantitySeparationMetricName+'-either.pkl','wb') as f:
            pickle.dump(quantitySeparationDf,f)
    else:
        with open('kineticFeatureOutput/quantitySeparationMetric-'+folderName+'-'+quantitySeparationMetricName+'.pkl','wb') as f:
            pickle.dump(quantitySeparationDf,f)
