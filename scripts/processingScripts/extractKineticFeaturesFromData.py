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

#Note: ADD NEW KINETIC FEATURES HERE; MAKE SURE THEY ONLY REQUIRE THE TIMESLICED DATAFRAME AND THE LIST OF OBSERVABLES AS INPUT PARAMETERS AND ONLY RETURN
#A DATAFRAME WITH THE KINETIC FEATURE
kineticFeatureDictionary = {'SlopesYIntercepts':createLinRegDataFrame,'Sums':createSumDataFrame}

#When given a dataframe and a minimum timepoint range, return all possible timeslices (regions of the timeseries) that have at least 
#minTimepointScaleFactor*totalNumberTimepoints datapoints within them. Helps reduce noise by not calculating kinetic features on very sparse
#regions of the time kinetics graph
def returnTimePointEndpoints(df,minTimePointScaleFactor):
    #Get all timepoints in dataframe
    timepoints = list(df.columns)
    #Get all combinations of the indices of the timepoints
    timepointEndpoints = list(itertools.combinations(range(len(timepoints)),2))
    timepointRegionList = []
    minTimePointRegionLength = minTimePointScaleFactor*len(timepoints)
    for timepointEndpoint in timepointEndpoints:
        #Grab all timepoints between a combination of timepoint indices
        timepointRegion = timepoints[timepointEndpoint[0]:timepointEndpoint[1]]
        #If length of this timepoint region is larger than the minimum, add both the starting and ending timepoints to a list
        if len(timepointRegion) >= minTimePointRegionLength:
            timepointRegionList.append(tuple([timepointRegion[0],timepointRegion[-1]]))
    return timepointRegionList

#Time partitions data by timepoint region for each observable, concatenates into statistic df, returns list of statistic dfs to be made into cellTypeDf
def returnFeatureDataStatisticList(inputStatisticDf,dataType,minTimePointScaleFactor):
    featureStatisticDfList = []
    statisticList = list(pd.unique(inputStatisticDf.index.get_level_values('Statistic')))
    for statistic in statisticList:
        #Make statistic sliced dataframes (only one dummy statistic for cytokines/proliferation, many statistics (GFI, CV, % Positive etc.) for cells
        featureStatisticDf = inputStatisticDf.xs([statistic],level=['Statistic'])
        if dataType == 'cyt':
            observableName = 'Cytokine'
        elif dataType == 'cell':
            observableName = 'Marker'
        else:
            observableName = 'Metric'

        ###Add individual timepoints of each observable in the datatype as features, primarily to compare their deconvolution against the best kinetic features
        #(to make the point that we need the time aspect of the data to get good deconvolution)###
        #Unstack Observable (move observable to columns; columns now have time-observable))
        individualTimepointDfToReindex = featureStatisticDf.unstack(observableName)
        #Grab a dataframe containing the first observable from the statisticDf
        reindexingDf = featureStatisticDf.xs([list(pd.unique(featureStatisticDf.index.get_level_values(observableName)))[0]],level=[observableName]) 
        #Unstacking automatically sorts the dataframe in lexographic order. We use this method to recover the original ordering of the index
        individualTimepointDfBeforeNewColumns = reindexDataFrame(individualTimepointDfToReindex,reindexingDf,False)
        #Make new column index for individualtimepoint df
        newDfList = []
        timeslicelist = []
        #Go through each timepoint and observable and construct a list containing the timepoint as the timeslicestart and end, the observable as the observable,
        #and the feature type as "Individual Observation" (allows for easy subsetting later on)
        for timepoint in pd.unique(individualTimepointDfBeforeNewColumns.columns.get_level_values('Time')):
            currentTimeDf = individualTimepointDfBeforeNewColumns.loc[:,timepoint]
            for observable in currentTimeDf:
                timeslicelist.append([timepoint,timepoint,'IndividualObservation',observable])
        #Construct new column multindex and dataframe with previously constructed list
        newMultiIndexColumns = pd.MultiIndex.from_tuples(timeslicelist,names=['TimeSliceStart','TimeSliceEnd','FeatureType','Observable'])
        individualTimepointDf = pd.DataFrame(individualTimepointDfBeforeNewColumns.values,index=individualTimepointDfBeforeNewColumns.index,columns=newMultiIndexColumns)
        
        #Grab all "timepoint regions" possible for the timeseries (5-20 hours, 5-25 hours etc.) and start iterating through them
        timepointRegions = returnTimePointEndpoints(featureStatisticDf,minTimePointScaleFactor)
        timepointRegionDfList = []
        for timepointRegion in timepointRegions:
            timeStart = timepointRegion[0]
            timeEnd = timepointRegion[1]
            timeStartIndex = list(featureStatisticDf.columns).index(timeStart)
            timeEndIndex = list(featureStatisticDf.columns).index(timeEnd)
            
            #Get all observables for this datatype and statistic (doesn't change in cyt/prolif, but does change per statistic in cells)
            observableList = list(pd.unique(featureStatisticDf.index.get_level_values(observableName)))
            #Slice the time kinetics data into specified region
            df = featureStatisticDf.iloc[:,timeStartIndex:timeEndIndex+1]
            #Start calculating kinetic features from kinetic feature dictionary, and start adding the returned dataframes to a list
            kineticFeatureList = []
            for kineticFeature in kineticFeatureDictionary:
                kineticFeatureDf = kineticFeatureDictionary[kineticFeature](df,observableList)
                kineticFeatureList.append(kineticFeatureDf)
            #Combine all kinetic features for all observables for a particular time slice into single dataframe
            timepointRegionFeatureDf = pd.concat(kineticFeatureList,axis=1)
            timepointRegionDfList.append(timepointRegionFeatureDf)
            print('\t\t\t'+str(timeStart)+'hrs-'+str(timeEnd)+'hrs done!')
        #all feature dataframes for all observables in a statistic get concatenated into a single dataframe
        featureStatisticMultiIndex = pd.MultiIndex.from_tuples(timepointRegions,names=['TimeSliceStart','TimeSliceEnd'])
        featureStatisticDf = pd.concat(timepointRegionDfList,axis=1,keys=timepointRegions,names=['TimeSliceStart','TimeSliceEnd'])
        #The kinetic feature values and the invidual timepoint df constructed earlier are joined columnwise to produce the final feature df for the statistic
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
    #Grab all pickle files in kineticFeatureInputFolder, process kinetic features for each datafarme one at a time
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
        #Cytokine and proliferation dataframes have essentially the same structure, while
        #cell dataframes actually use the celltype and statistic columns
        if dataType != 'cell':
            #Only one dummy celltype/statistic for cytokine/proliferation data
            #Cytokine Data
            if dataType == 'cyt':
                cellTypeList = ['NotApplicable']
                featureStatisticList = ['CytokineConcentration']
                #Need to take logarithm of cytokine data for features to work
                dataTypeDf = np.log10(dataTypeDf)
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
            #Need to swap marker and statistic levels for the dataframe to align with the desired formats
            #Before flipping, the first three levels in the cell df go cellType-Observable-Statistic, but we want observable
            #on the outside so we swap levels 1 and 2
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
        #all celltype dataframes concatenated into a dataType dataframe
        dataTypeDf = pd.concat(cellTypeDfList,axis=1,keys=cellTypeList,names=['CellType'])
        dataTypeDfList.append(dataTypeDf)
        print(str(dataType)+' done!')
    #all dataType dataframes concatenated into a full kinetic feature df for the experiment
    fullFeatureDf = pd.concat(dataTypeDfList,axis=1,keys=dataTypeList,names=['DataType'])
    #Save Raw Features (will be preprocessed in later script)
    with open('kineticFeatureOutput/fullFeatureDf-'+folderName+'-raw.pkl','wb') as f:
        pickle.dump(fullFeatureDf,f)
    return fullFeatureDf
