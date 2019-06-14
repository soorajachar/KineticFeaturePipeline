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
    #sns.set(font_scale=2,rc={'figure.figsize':(7.7,7.5)})
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
        #numericPeptideList.append(len(peptideList)-1-peptideList.index(peptide))
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
    #plt.legend(loc='center left',bbox_to_anchor=(1,0.5))
    #plt.show()
    plt.savefig('kineticFeatureFigures/top'+str(numberOfTopScoring)+'-quantityMetrics-'+folderName+'-'+dataTypeString+'.png',bbox_inches='tight')
