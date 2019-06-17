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
from sklearn.preprocessing import MinMaxScaler
import math,pickle,os,sys,fcsparser,json,time,glob,itertools,subprocess,argparse
sys.path.insert(0, 'processingScripts/')
from extractKineticFeaturesFromData import createFullFeatureDataFrame
from deconvolutionMetricProcessing import qualityMetric,quantityMetric
from miscFunctions import parseCommandLineNNString
#Preprocessing
from preprocessKineticFeatures import preprocessingPipeline 
#Figure Generation
sys.path.insert(0, 'figureScripts/')
from createDeconvolutionMetricPlots import createQualityStripPlot,createQuantityCatPlot,createQualityQuantityPairPlot,createQualityKDE
sys.path.insert(0, 'facetPlottingScripts/')
import facetPlotLibrary as fpl

idx = pd.IndexSlice

#Should have experiment spreadsheet in this folder
pathToExperimentsFolder = '../experimentDataKineticFeatures'

#Subfolders created within experiment folder (inputData for cell/cyt/prolif dataframes, output data for all kinetic feature/mutual info dataframes
#figures for all generated plots
inputDataSubfolderName = 'kineticFeatureInput'
outputDataSubfolderName = 'kineticFeatureOutput'
figureSubfolderName = 'kineticFeatureFigures'

#Top X scoring plots to plot
numberOfTopScoring = 10

#Distance metrics to use for assessing quality and quantity deconvolution
qualitySeparationMetricName = 'mutualInformation'
quantitySeparationMetricName = 'CV'

#Make experiment folder and subfolders
def createExperimentFolders(experimentFolderName):
    subprocess.run(['mkdir',pathToExperimentsFolder+'/'+experimentFolderName])
    
    experimentSubfolderNames = [inputDataSubfolderName,outputDataSubfolderName,figureSubfolderName]
    for experimentSubfolderName in experimentSubfolderNames:
        subprocess.run(['mkdir',pathToExperimentsFolder+'/'+experimentFolderName+'/'+experimentSubfolderName])

def main():
    parser = argparse.ArgumentParser(description="Process and plot kinetic features.")
    parser.add_argument("-ce", action='store_true', help = "Create experiment folder and subfolders for storing kinetic data input/output/figures.")
    parser.add_argument("-pd", action='store_true', help = "Process Kinetic Feature FeatureDf.")
    parser.add_argument("-ppd", action='store_true', help = "Process Kinetic Feature FeatureDf.")
    parser.add_argument("-dc", action='store_true', help = "Process Kinetic Feature Deconvolution Metrics.")
    parser.add_argument("-pp", action='store_true', help = "Plot Pair Plot of kinetic features (used for quality and quantity).")
    parser.add_argument("-cp", action='store_true', help = "Plot Cat Plots of kinetic features (used for quantity or quantity).")
    parser.add_argument("-kde", action='store_true', help = "Plot KDE of kinetic features (used for quality).")
    parser.add_argument("-bp", action='store_true', help = "Plot Bar Plot of kinetic features")
    parser.add_argument("-sp", action='store_true', help = "Plot Scatter Plot of kinetic features")
    parser.add_argument("--input", dest='inputString', help ="Run specified script on these experimental data set(s). Separate numbers with , or - for a range.")
    
    parser.add_argument("-cyt", action='store_true', help = "Process cytokine data type.")
    parser.add_argument("-cell", action='store_true', help = "Process cell data type.")
    parser.add_argument("-prolif", action='store_true', help = "Process proliferation data type.")
    parser.add_argument("-all", action='store_true', help = "Process all data types.")
    
    parser.add_argument("-qual", action='store_true', help = "Plot quality")
    parser.add_argument("-quan", action='store_true', help = "Plot quantity")
    
    args = parser.parse_args()

    #Allows for multiple datasets to be run at once using , or - with experiment numbers
    experimentsToRun = parseCommandLineNNString(args.inputString)
    excel_data = pd.read_excel(pathToExperimentsFolder+'/masterExperimentSpreadsheet.xlsx')

    #If specific datatype (cell/prolif/cyt) not specified, run script considering all datatypes
    #Mostly useful to exclude cell, as it has extra levels (celltype and statistic) that are 
    #not used in the other data types
    if args.all or not (args.cyt or args.cell or args.prolif):
        dataTypeList = slice(None)
        dataTypeString = 'all'
    else: 
        dataTypeList = []
        dataTypeInputs = {'cyt':args.cyt,'cell':args.cell,'prolif':args.prolif}
        for dataType in dataTypeInputs:
            if dataTypeInputs[dataType]:
                dataTypeList.append(dataType)
        dataTypeString = '_'.join(dataTypeList)
    #Go through every experiment number passed through --input, grab the experiment name from the spreadsheet, and run script on that experiment
    for expNum in experimentsToRun:
        folderName = excel_data['Full Name'][expNum-1]
        os.chdir(pathToExperimentsFolder)
        if not args.ce:
            os.chdir(folderName)
            if not args.pd:
                #Load in whole kinetic feature dataframe
                rawFeatureDf = pickle.load(open('kineticFeatureOutput/fullFeatureDf-'+folderName+'-raw.pkl','rb'))
                if not args.ppd:
                    #Load in preprocessed dataframes. "Quality Ordered" dataframe only has features that arrange qualities in the correct order
                    #"Quantity Ordered" dataframe only has features that arrange quantity in the correct order, so "either ordered" only has dataframes that
                    #arrange at least one of the two metrics in the correct order. Also excludes features that give the same value for multiple peptides/concentrations
                    qualityOrderedFeatureDf = pickle.load(open('kineticFeatureOutput/fullFeatureDf-'+folderName+'-preprocessed-qualityOrdered.pkl','rb'))
                    quantityOrderedFeatureDf = pickle.load(open('kineticFeatureOutput/fullFeatureDf-'+folderName+'-preprocessed-quantityOrdered.pkl','rb'))
                    eitherOrderedFeatureDf = pickle.load(open('kineticFeatureOutput/fullFeatureDf-'+folderName+'-preprocessed-either.pkl','rb'))
                
       #Processing
       #Create kinetic feature folders for experiment in spreadsheet to put input/output data/figures in (need to do this first before using any other methods)
        if args.ce:
            createExperimentFolders(folderName)
        #Extract kinetic features from all input dataframes placed in kineticFeatureInput dataframes in experiment folder
        #Can change number of timepoints used for timeslices with "minTimePointScaleFactor" parameter (default is that minimum
        #timeslice length is 25% of total number of timepoints in data)
        elif args.pd:
            minTimePointScaleFactor = 0.25
            rawFeatureDf = createFullFeatureDataFrame(expNum,folderName,minTimePointScaleFactor)
            preprocessingPipeline(rawFeatureDf,expNum,folderName)
        elif args.ppd:
            preprocessingPipeline(rawFeatureDf,expNum,folderName)
        #Create mutual info and cv metrics for measuring quality and quantity separation respectively. Creates dataframes for both individually sorted and both sorted feature dataframes
        elif args.dc:
            #Create distance metric (quality) or mutual info metric (quality2) df
            qualityMetric(qualityOrderedFeatureDf,expNum,folderName,qualitySeparationMetricName,False)
            qualityMetric(eitherOrderedFeatureDf,expNum,folderName,qualitySeparationMetricName,True)
            #Create cv metric (quantity) df
            quantityMetric(quantityOrderedFeatureDf,expNum,folderName,quantitySeparationMetricName,False)
            quantityMetric(eitherOrderedFeatureDf,expNum,folderName,quantitySeparationMetricName,True)
        #Figure Generation
        elif args.pp or args.cp or args.sp or args.bp or args.kde:
            qualityMetricDf = pickle.load(open('kineticFeatureOutput/qualitySeparationMetric-'+folderName+'-'+qualitySeparationMetricName+'.pkl','rb')).loc[idx[dataTypeList]]
            quantityMetricDf = pickle.load(open('kineticFeatureOutput/quantitySeparationMetric-'+folderName+'-'+quantitySeparationMetricName+'.pkl','rb')).loc[idx[dataTypeList]]
            qualityMetricEitherOrderedDf = pickle.load(open('kineticFeatureOutput/qualitySeparationMetric-'+folderName+'-'+qualitySeparationMetricName+'-either.pkl','rb')).loc[idx[dataTypeList]]
            quantityMetricEitherOrderedDf = pickle.load(open('kineticFeatureOutput/quantitySeparationMetric-'+folderName+'-'+quantitySeparationMetricName+'-either.pkl','rb')).loc[idx[dataTypeList]]
            #Pair plot; useful for plotting all possible 2d relationships between features good at deconvolving quality and those good at deconvolving quantity
            if args.pp:
                createQualityQuantityPairPlot(qualityMetricDf,quantityMetricDf,eitherOrderedFeatureDf.loc[:,idx[dataTypeList]],numberOfTopScoring,folderName,dataTypeString)
            #Catplots; useful for plotting 1d plots of a feature to show how well it can deconvolve either quality or quantity separately
            elif args.cp:
                if args.qual:
                    createQualityStripPlot(qualityMetricDf,qualityOrderedFeatureDf,numberOfTopScoring,folderName,dataTypeString)
                elif args.quan:
                    createQuantityCatPlot(quantityMetricDf,quantityOrderedFeatureDf,numberOfTopScoring,folderName,dataTypeString)
                else:
                    createStripPlot(qualityMetricDf,qualityOrderedFeatureDf,numberOfTopScoring,folderName,dataTypeString)
                    createQuantityCatPlot(quantityMetricDf,quantityOrderedFeatureDf,numberOfTopScoring,folderName,dataTypeString)
            #KDEs; useful for showing quality deconvolution
            elif args.kde:
                createQualityKDE(qualityMetricDf,qualityOrderedFeatureDf,numberOfTopScoring,folderName,dataTypeString)
            elif args.sp or args.bp:
                if args.sp:
                    subPlotType = 'scatter'
                    plotType = 'ordered'
                else:
                    subPlotType = 'bar'
                    plotType = 'categorical'
                
                dataType=dataTypeString
                combined_df = pd.concat([quantityMetricEitherOrderedDf,qualityMetricEitherOrderedDf],axis=1)
                combined_df.columns.name = 'DeconvolutionParameters'
                useModifiedDf = True
                plt.switch_backend('QT4Agg') #default on my system
                fpl.facetPlottingGUI(combined_df,plotType,subPlotType,dataType)
                subsettedDfList,subsettedDfListTitles,figureLevels,levelValuesPlottedIndividually = fpl.produceSubsettedDataFrames(combined_df)
                fpl.plotFacetedFigures(folderName,plotType,subPlotType,dataType,subsettedDfList,subsettedDfListTitles,figureLevels,levelValuesPlottedIndividually,useModifiedDf,combined_df)
        else:
            sys.exit(0)
        if not args.ce:
            os.chdir('..')
        print(folderName + ' finished')
        os.chdir('../scripts/')
main()
