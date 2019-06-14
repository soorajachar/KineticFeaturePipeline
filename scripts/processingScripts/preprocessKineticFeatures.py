#!/usr/bin/env python3
import pandas as pd
import seaborn as sns
import numpy as np
import pickle,math,sys,re
from sklearn.preprocessing import MinMaxScaler

canonicalQualityOrder = ['N4','Q4','T4','A8','Q7','V4','G4','E1']

#Remove peptides/concentrations that are low enough to not provoke a response from the Tcells
#Should think of a way to do automatically and objectively
def extractPeptidesConcentrationsWithResponse(standardizedFeatureDf,expNum):
    #N4 10pM and T4/A8 100pM did not provoke a response from the TCells
    if expNum == 93:
        standardizedFeatureDf = standardizedFeatureDf.iloc[[0,1,2,3,5,6,7,8,10,11,12,13,15],:]
    #Q4 100 pM did not elicit a response
    elif expNum == 96:
        standardizedFeatureDf = standardizedFeatureDf.iloc[[0,1,2,3,4,5,6,8,9,10,11,12],:]
    else:
        pass
    return standardizedFeatureDf

#Preprocess to only use markers which behaved well in the experiment; not strictly needed but can help save checking to see if a kinetic feature for a particular marker is meaningful
#Need to think of a way to discard markers which have obvious batch effect (should be needed much less now that we've switched to fixation without permeablization)
def returnWellBehavedMarkers(expNum):
    #specify working markers per experiment
    if expNum == 93:
        markers = ['CD25','CD27','CD45RB','IRF4','TBET','CTV','CTFR']
    elif expNum == 96:
        markers = ['CD25','CD27','IRF4','TBET']
    elif expNum == 117:
        markers = ['CD27','CD25','CD3e','CD62L','PDL1','PD1','CD54','CD44','CD45']
    #if not specified; use all markers
    else:
        markers = slice(None)
    #Needed for counts/percents of specific populations (which have no particular marker)
    if markers != slice(None):
        markers.append('NotApplicable')
    else:
        pass
    return markers

#Should think about what normalization method is best to use in this case
def normalizeFeatureDataFrame(scalingType,df):
    if scalingType == 'minmax':
        scaler = MinMaxScaler()
        standardizedDf = pd.DataFrame(scaler.fit_transform(df),index=df.index,columns=df.columns)
    return standardizedDf

#Removes features that do not preserve concentration order correctly
def checkQuantityOrder(featureDf):
    quantityOrderedFeatures = []
    for i in range(featureDf.shape[1]):
        observableDf = featureDf.iloc[:,i]
        allOrdered = True
        #Take advantage of the fact that the order of concentrations of a given peptide is always descending
        #Use the original order of concentrations in a peptide, check to see if this matches ascending/descending concentrations
        #which would mean that the feature does sort concentrations correctly
        for peptide in list(pd.unique(observableDf.index.get_level_values('Peptide'))):
            concentrationValues = list(observableDf.loc[peptide])
            sortedAscendingConcentrationValues = sorted(concentrationValues)
            sortedDescendingConcentrationValues = sorted(concentrationValues)[::-1]
            #establish direction of sorting with first peptide
            if peptide == list(pd.unique(observableDf.index.get_level_values('Peptide')))[0]:
                #Also need to check that direction of concentration order is conserved across all peptides of a given feature
                #e.g. if an increase in a feature results in an increase in N4 concentration, an increase in the same feature
                #should not result in a decrease in the T4 concentration
                if concentrationValues == sortedAscendingConcentrationValues:
                    ascending = True
                else:
                    if concentrationValues == sortedDescendingConcentrationValues:
                        ascending = False
                    else:
                        allOrdered = False
                        break
            else:
                if ascending:
                    if concentrationValues != sortedAscendingConcentrationValues:
                        allOrdered = False
                        break
                else:
                    if concentrationValues != sortedAscendingConcentrationValues:
                        allOrdered = False
                        break
        if allOrdered:
            quantityOrderedFeatures.append(i)
            print('Feature '+str(i)+' sorts  quantity correctly')
    return featureDf.iloc[:,quantityOrderedFeatures]

def checkQualityOrder(featureDf):
    qualityOrderedFeatures = []
    #peptides = pd.unique(featureDf.index.get_level_values('Peptide'))
    for i in range(featureDf.shape[1]):
        observableDf = featureDf.iloc[:,i]
        peptideValues = observableDf.groupby(['Peptide']).mean()
        sortedPeptideValuesAscending = list(pd.unique(peptideValues.sort_values().index))
        sortedPeptideValuesDescending = list(pd.unique(peptideValues.sort_values(ascending=False).index))
        peptideQualities = []
        for peptide in canonicalQualityOrder:
            if peptide in list(pd.unique(observableDf.index.get_level_values('Peptide'))):
                peptideQualities.append(peptide)
        if peptideQualities == sortedPeptideValuesAscending or peptideQualities == sortedPeptideValuesDescending:
            qualityOrderedFeatures.append(i)
            print('Feature '+str(i)+' sorts quality correctly')
    return featureDf.iloc[:,qualityOrderedFeatures]

def preprocessingPipeline(fullFeatureDf,expNum,folderName):
    #Fill any nans with 0 (need to think about what to do here; maybe drop the feature if it has nans instead?
    fullFeatureDf.fillna(0,inplace=True)
    #Remove peptides that do not provoke a response
    respondingPeptides = extractPeptidesConcentrationsWithResponse(fullFeatureDf,expNum)
    #Normalize remaining peptides (with min max for now)
    normalizedFeatureDf = normalizeFeatureDataFrame('minmax',respondingPeptides)
    with open('kineticFeatureOutput/fullFeatureDf-'+folderName+'-preprocessed.pkl','wb') as f:
        pickle.dump(normalizedFeatureDf,f)
    #Prune features; features that do not correctly sort quality or quantity are thrown out
    correctlyOrderedByQuality = checkQualityOrder(normalizedFeatureDf)
    correctlyOrderedByQuantity = checkQuantityOrder(normalizedFeatureDf)
    first_list = list(correctlyOrderedByQuality.columns)
    second_list = list(correctlyOrderedByQuantity.columns)
    sortsEitherCorrectly = first_list + list(set(second_list) - set(first_list))
    correctlyOrderedByEither = normalizedFeatureDf.loc[:,sortsEitherCorrectly]
    with open('kineticFeatureOutput/fullFeatureDf-'+folderName+'-preprocessed-qualityOrdered.pkl','wb') as f:
        pickle.dump(correctlyOrderedByQuality,f)
    with open('kineticFeatureOutput/fullFeatureDf-'+folderName+'-preprocessed-quantityOrdered.pkl','wb') as f:
        pickle.dump(correctlyOrderedByQuantity,f)
    with open('kineticFeatureOutput/fullFeatureDf-'+folderName+'-preprocessed-either.pkl','wb') as f:
        pickle.dump(correctlyOrderedByEither,f)
