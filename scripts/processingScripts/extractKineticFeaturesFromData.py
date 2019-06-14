#!/usr/bin/env python3
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
import math,pickle,os,sys,fcsparser,json,time,glob,itertools,subprocess,argparse
#Kinetic Features MAKE SURE TO IMPORT THEM FROM KINETIC FEATURE LIBRARY IF YOU ADD NEW ONES
from kineticFeatureLibrary import createLinRegDataFrame,createSumDataFrame
from miscFunctions import sortSINumerically,reindexDataFrame,parseCommandLineNNString
from preprocessKineticFeatures import preprocessingPipeline,returnWellBehavedMarkers

idx = pd.IndexSlice

#Note: ADD NEW KINETIC FEATURES HERE
kineticFeatureDictionary = {'SlopesYIntercepts':createLinRegDataFrame,'Sums':createSumDataFrame}

def returnTimePointEndpoints(df,minTimePointScaleFactor):
    timepoints = list(df.columns)
    timepointEndpoints = list(itertools.combinations(range(len(timepoints)),2))
    timepointRegionList = []
    minTimePointRegionLength = minTimePointScaleFactor*len(timepoints)
    for timepointEndpoint in timepointEndpoints:
        timepointRegion = timepoints[timepointEndpoint[0]:timepointEndpoint[1]]
        if len(timepointRegion) >= minTimePointRegionLength:
            timepointRegionList.append(tuple([timepointRegion[0],timepointRegion[-1]]))
    return timepointRegionList

#Time partitions data by timepoint region for each observable, concatenates into statistic df, returns list of statistic dfs to be made into cellTypeDf
def returnFeatureDataStatisticList(inputStatisticDf,dataType,minTimePointScaleFactor):
    featureStatisticDfList = []
    statisticName = 'Statistic'
    statisticList = list(pd.unique(inputStatisticDf.index.get_level_values('Statistic')))
    for statistic in statisticList:
        #Make statistic sliced dataframes (only one dummy statistic for cytokines/proliferation, many statistics (GFI, CV, % Positive etc.) for cells
        featureStatisticDf = inputStatisticDf.xs([statistic],level=[statisticName])
        #Use Individual Timepoints
        if dataType == 'cyt':
            observableName = 'Cytokine'
        elif dataType == 'cell':
            observableName = 'Marker'
        else:
            observableName = 'Metric'
        individualTimepointDfToReindex = featureStatisticDf.unstack(observableName)
        reindexingDf = featureStatisticDf.xs([list(pd.unique(featureStatisticDf.index.get_level_values(observableName)))[0]],level=[observableName]) 
        reindexedDf = reindexDataFrame(individualTimepointDfToReindex,reindexingDf,False)
        newDfList = []
        timeslicelist = []
        for timepoint in pd.unique(reindexedDf.columns.get_level_values('Time')):
            currentTimeDf = reindexedDf.loc[:,timepoint]
            newDfList.append(currentTimeDf)
            timeslicelist.append([timepoint,timepoint])
        individualTimepointDf = pd.concat(newDfList,axis=1,keys=pd.unique(reindexedDf.columns.get_level_values('Time')),names=['TimeSliceEnd'])
        timeslicelist = []
        for timepoint in pd.unique(reindexedDf.columns.get_level_values('Time')):
            currentTimeDf = individualTimepointDf.loc[:,timepoint]
            newDfList.append(currentTimeDf)
            for observable in currentTimeDf:
                timeslicelist.append([timepoint,timepoint,'IndividualObservation',observable])
        newMultiIndexColumns = pd.MultiIndex.from_tuples(timeslicelist,names=['TimeSliceStart','TimeSliceEnd','FeatureType','Observable'])
        individualTimepointDf = pd.DataFrame(individualTimepointDf.values,index=individualTimepointDf.index,columns=newMultiIndexColumns)
        #Grab all "timepoint regions" possible for the timeseries (5-20 hours, 5-25 hours etc.) and start iterating through them
        timepointRegions = returnTimePointEndpoints(featureStatisticDf,minTimePointScaleFactor)
        timepointRegionDfList = []
        for timepointRegion in timepointRegions:
            timeStart = timepointRegion[0]
            timeEnd = timepointRegion[1]
            timeStartIndex = list(featureStatisticDf.columns).index(timeStart)
            timeEndIndex = list(featureStatisticDf.columns).index(timeEnd)
            
            observableList = list(pd.unique(featureStatisticDf.index.get_level_values(observableName)))
            #Slice the time kinetics data into specified region
            df = featureStatisticDf.iloc[:,timeStartIndex:timeEndIndex+1]
            kineticFeatureList = []
            for kineticFeature in kineticFeatureDictionary:
                kineticFeatureDf = kineticFeatureDictionary[kineticFeature](df,observableList)
                kineticFeatureList.append(kineticFeatureDf)
            #Combine all kinetic features for all observables for a particular time slice into single dataframe
            timepointRegionFeatureDf = pd.concat(kineticFeatureList,axis=1)
            timepointRegionDfList.append(timepointRegionFeatureDf)
            print('\t\t\t'+str(timeStart)+'hrs-'+str(timeEnd)+'hrs done!')
        #all feature dataframes for all observables in a statistic
        featureStatisticMultiIndex = pd.MultiIndex.from_tuples(timepointRegions,names=['TimeSliceStart','TimeSliceEnd'])
        featureStatisticDf = pd.concat(timepointRegionDfList,axis=1,keys=timepointRegions,names=['TimeSliceStart','TimeSliceEnd'])
        featureStatisticDfWithIndividualTimepoints = pd.concat([featureStatisticDf,individualTimepointDf],axis=1)
        featureStatisticDfList.append(featureStatisticDfWithIndividualTimepoints)
        print('\t\t'+str(statistic)+' done!')
    return featureStatisticDfList

#Preprocess statistic df to allow kinetic feature method to operate on it
def returnKineticFeatureDataFrameSubset(fulldf,expNum,dataType):
    #Extract relevant cytokines/statistics/markers from statistic df. Marker relevance can vary per experiment, while other two remain the same across all experiments
    if dataType == 'cyt':
        subsettedDf = fulldf.loc[idx[['IFNg','IL-2','IL-6','TNFa']],:]
    elif dataType == 'prolif':
        subsettedDf = fulldf.loc[idx[['proliferationIndex','divisionIndex','fractionDiluted','precursorFrequency']],:]
    else:
        desiredMarkerList = returnWellBehavedMarkers(expNum)
        subsettedDf = fulldf.loc[idx[:,desiredMarkerList],:]
    #Preprocess to remove all remaining levels in dataframes besides Observable/Peptide/Concentration (only needed for some experiments; based on experiment number)
    if expNum == 93:
        subsettedDf = subsettedDf.xs(['WT'],level=['Genotype'])
    else:
        pass
    return subsettedDf

def createFullFeatureDataFrame(expNum,folderName,minTimePointScaleFactor):
    dataTypeDfList = []
    dataTypeList = []
    
    os.chdir('kineticFeatureInput')
    dataList =  glob.glob('*.pkl')
    os.chdir('..')
    for fileName in dataList:
        #Determine dataType from file prefix
        if 'cytokineConcentration' in fileName:
            dataType = 'cyt'
        elif 'cellStatistic' in fileName:
            dataType = 'cell'
        else:
            dataType = 'prolif'
        dataTypeList.append(dataType)
        
        dataTypeDf = pickle.load(open('kineticFeatureInput/'+fileName,'rb'))
        cellTypeDfList = []
        if dataType != 'cell':
            #Only one dummy celltype/statistic for cytokine/proliferation data
            #Cytokine Data
            if dataType == 'cyt':
                cellTypeList = ['NotApplicable']
                featureStatisticList = ['CytokineConcentration']
                dataTypeDf = abs(np.log10(dataTypeDf))
            #Proliferation Data
            else:
                cellTypeList = ['TCells']
                featureStatisticList = ['ProliferationMetric']
                tempTuple = []
                for row in range(dataTypeDf.shape[0]):
                    tempTuple.append(dataTypeDf.iloc[row,:].name)
                newMI = pd.MultiIndex.from_tuples(tempTuple,names=['Metric']+dataTypeDf.index.names[1:])
                dataTypeDf = pd.DataFrame(dataTypeDf.values,index=newMI,columns=dataTypeDf.columns)

            #Grab subsetted df, then grab statistic df (only a single entry) then add to celltype df (only a single entry)
            inputStatisticDf = returnKineticFeatureDataFrameSubset(dataTypeDf,expNum,dataType)
            inputStatisticDf = pd.concat([inputStatisticDf],keys=featureStatisticList,names=['Statistic'])
            featureStatisticDfList = returnFeatureDataStatisticList(inputStatisticDf,dataType,minTimePointScaleFactor)
            cellTypeDf = pd.concat(featureStatisticDfList,axis=1,keys=featureStatisticList,names=['Statistic'])
            cellTypeDfList.append(cellTypeDf)
        else:
            dataTypeDf = dataTypeDf.swaplevel(1,2)
            #Grab all unique cell types from input data and start iterating through them
            cellTypeList = list(pd.unique(dataTypeDf.index.get_level_values('CellType'))) 
            for cellType in cellTypeList:
                allStatisticDf = dataTypeDf.xs([cellType],level=['CellType'])
                #Grab only WT OT1 Data
                allStatisticDf = returnKineticFeatureDataFrameSubset(allStatisticDf,expNum,dataType)
                #Grab all unique statistics from current cell type and start iterating through them
                statisticList = list(pd.unique(allStatisticDf.index.get_level_values('Statistic'))) 
                statisticDfList = returnFeatureDataStatisticList(allStatisticDf,dataType,minTimePointScaleFactor)
                #Start combining feature dataframes:
                #all statistic dfs into a celltype
                cellTypeDf = pd.concat(statisticDfList,axis=1,keys=statisticList,names=['Statistic'])
                cellTypeDfList.append(cellTypeDf)
                print('\t'+str(cellType)+' done!')
        #all celltypes into a dataType
        dataTypeDf = pd.concat(cellTypeDfList,axis=1,keys=cellTypeList,names=['CellType'])
        dataTypeDfList.append(dataTypeDf)
        print(str(dataType)+' done!')
    #all dataTypes into a full kinetic feature df for the experiment
    fullFeatureDf = pd.concat(dataTypeDfList,axis=1,keys=dataTypeList,names=['DataType'])
    #Raw Features
    with open('kineticFeatureOutput/fullFeatureDf-'+folderName+'-raw.pkl','wb') as f:
        pickle.dump(fullFeatureDf,f)
    return fullFeatureDf
