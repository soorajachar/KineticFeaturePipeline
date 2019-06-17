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
sys.path.insert(0, '../processingScripts/')
from miscFunctions import sortSINumerically
idx = pd.IndexSlice

#Returns the actual values of all features in the plottingSubset dataframe from the original featureDf, along with a list of
#modified column names these column names will also be used to construct a new dataframe which only has the feature name as a column (all of the levels
#are concatenated together)
def grabTopScoringFeatureData(plottingSubset,featureDf):
    mi_df = []
    tupleList = []
    plottingDfList = []
    #Go through each row in the plotting subset dataframe (which has feature levels as its index and no real columns)
    for row in range(plottingSubset.shape[0]):
        currentrow = plottingSubset.iloc[row,:].name
        #Successively subset each dataframe with each level in the feature df. Need to do it this way to remove the previous levels
        #and be left with a single level dataframe (as opposed to just using .loc[:,currentrow] immediately, which preserves the subsetted levels
        featureDf2 = featureDf.copy()
        for indexer in currentrow:
            featureDf2 = featureDf2.loc[:,indexer]
        plottingDfList.append(featureDf2)
        tupleList.append(currentrow)
    #Create new dataframe with only single level in columns (and standard peptide/concentration index) that is the featurename
    plottingColumnsBeforeModification = pd.MultiIndex.from_tuples(tupleList,names=featureDf.columns.names)
    plottingColumns = []
    #\n escape characters are added at the end of each level to keep the tick label length reasonable
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

def createQualityQuantityPairPlot(qualityMetricDf,quantityMetricDf,featureDf,numberOfTopScoring,folderName,dataTypeString):
    sortedQualityMetricDf = qualityMetricDf.sort_values('Quality Separation',ascending=False)
    sortedQuantityMetricDf = quantityMetricDf.sort_values('Quantity Separation',ascending=False)
    plottingSubsetQuality = sortedQualityMetricDf.iloc[:numberOfTopScoring,:]
    plottingSubsetQuantity = sortedQuantityMetricDf.iloc[:numberOfTopScoring,:]
    plottingColumnsQuality,plottingDfListQuality = grabTopScoringFeatureData(plottingSubsetQuality,featureDf)
    plottingColumnsQuantity,plottingDfListQuantity = grabTopScoringFeatureData(plottingSubsetQuantity,featureDf)
    plottingDfQuality = pd.DataFrame(np.matrix(plottingDfListQuality).T,index=featureDf.index,columns=plottingColumnsQuality)
    plottingDfQuantity = pd.DataFrame(np.matrix(plottingDfListQuantity).T,index=featureDf.index,columns=plottingColumnsQuantity)
    plottingDf = pd.concat([plottingDfQuality,plottingDfQuantity],axis=1).reset_index()
    g = sns.pairplot(plottingDf,hue='Peptide',y_vars=plottingColumnsQuality,x_vars=plottingColumnsQuantity)
    for ax in g.fig.axes:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig('kineticFeatureFigures/top'+str(numberOfTopScoring)+'-qualityQuantityMetrics-'+folderName+'-'+dataTypeString+'.png',bbox_inches='tight')

def createQualityStripPlot(qualityMetricDf,featureDf,numberOfTopScoring,folderName,dataTypeString):
    sortedQualityMetricDf = qualityMetricDf.sort_values('Quality Separation',ascending=False)
    plottingSubsetQuality = sortedQualityMetricDf.iloc[:numberOfTopScoring,:]
    plottingColumnsQuality,plottingDfListQuality = grabTopScoringFeatureData(plottingSubsetQuality,featureDf)
    plottingDfQuality = pd.DataFrame(np.matrix(plottingDfListQuality).T,index=featureDf.index,columns=plottingColumnsQuality)
    plottingDfQuality.columns.name = 'Feature'
    plottingDf = plottingDfQuality.stack().to_frame('Feature Value').reset_index()
    g = sns.catplot(data=plottingDf,x='Feature',hue='Peptide',y='Feature Value',alpha=0.8,kind='strip')
    g.fig.set_size_inches(24,4)
    plt.savefig('kineticFeatureFigures/top'+str(numberOfTopScoring)+'-qualityMetrics-'+folderName+'-'+dataTypeString+'.png',bbox_inches='tight')

def createQualityKDE(qualityMetricDf,featureDf,numberOfTopScoring,folderName,dataTypeString):
    sortedQualityMetricDf = qualityMetricDf.sort_values('Quality Separation',ascending=False)
    plottingSubsetQuality = sortedQualityMetricDf.iloc[:numberOfTopScoring,:]
    plottingColumnsQuality,plottingDfListQuality = grabTopScoringFeatureData(plottingSubsetQuality,featureDf)
    plottingDfQuality = pd.DataFrame(np.matrix(plottingDfListQuality).T,index=featureDf.index,columns=plottingColumnsQuality)
    plottingDfQuality.columns.name = 'Feature'
    plottingDf = plottingDfQuality.stack().to_frame('Feature Value').reset_index()
    fg = sns.FacetGrid(plottingDf,sharey=False,col='Feature',hue='Peptide',col_wrap=int(numberOfTopScoring/2))
    fg.map(sns.kdeplot,'Feature Value',shade=True)
    plt.savefig('kineticFeatureFigures/top'+str(numberOfTopScoring)+'-qualityQuantityMetrics-kde-'+folderName+'-'+dataTypeString+'.png',bbox_inches='tight')

def createQuantityCatPlot(quantityMetricDf,featureDf,numberOfTopScoring,folderName,dataTypeString):
    sortedQuantityMetricDf = quantityMetricDf.sort_values('Quantity Separation',ascending=False)
    plottingSubsetQuantity = sortedQuantityMetricDf.iloc[:numberOfTopScoring,:]
    plottingColumnsQuantity,plottingDfListQuantity = grabTopScoringFeatureData(plottingSubsetQuantity,featureDf)
    plottingDfQuantity = pd.DataFrame(np.matrix(plottingDfListQuantity).T,index=featureDf.index,columns=plottingColumnsQuantity)
    plottingDfQuantity.columns.name = 'Feature'
    plottingDf = plottingDfQuantity.stack().to_frame('Feature Value').reset_index()
    numericPeptideList = []
    peptideList = list(pd.unique(plottingDf['Peptide']))
    for peptide in plottingDf['Peptide']:
        numericPeptideList.append(peptideList.index(peptide))
    plottingDf['Peptide'] = numericPeptideList
    so = sortSINumerically(pd.unique(plottingDf['Concentration']),True,True)[0]
    defaultPalette = sns.color_palette("tab10", len(peptideList))
    g = sns.relplot(data=plottingDf,y='Peptide',hue='Peptide',size='Concentration',size_order=so,col='Feature',x='Feature Value',alpha=0.8,kind='scatter',col_wrap = 10,palette=defaultPalette)#,facet_kws={'legend_out':True})
    g.fig.set_size_inches(30,4)
    for ax in g.fig.axes:
        ax.set_yticks([])
    newLabels = ['Peptide']+peptideList
    for t,l in zip(g._legend.texts[:len(peptideList)+1],newLabels):
        t.set_text(l)
    for ax in g.axes.flat:
        box = ax.get_position()
        ax.set_position([box.x0,box.y0,box.width*0.8,box.height])
    plt.savefig('kineticFeatureFigures/top'+str(numberOfTopScoring)+'-quantityMetrics-'+folderName+'-'+dataTypeString+'.png',bbox_inches='tight')
