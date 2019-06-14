#!/usr/bin/env python3
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import markers
import numpy as np
import seaborn as sns
import pandas as pd
import pickle,os,math,sys,itertools,re
from matplotlib.widgets import RadioButtons,Button,CheckButtons,TextBox

#Button width conserved across gui figures
buttonWidth = 0.1/2
buttonLength = 0.075/2
buttonXStart = 0.5-(0.01+buttonWidth)
buttonYStart = 0.01

def returnTicks(xticksToUse):
    logxticks = [-1000,-100,-10,0,10,100,1000,10000,100000]
    logicleXTicks = [64, 212, 229, 231, 233, 251, 399, 684, 925]
    xtickValues = []
    xtickLabels = []
    
    for logxtick in xticksToUse:
        if(logxtick < 0):
            xtickLabels.append('$-10^'+str(int(np.log10(-1*logxtick)))+'$')
        elif(logxtick == 0):
            xtickLabels.append('0')
        else:
            xtickLabels.append('$10^'+str(int(np.log10(logxtick)))+'$')
    
    for tickval in xticksToUse:
        xtickValue = logicleXTicks[logxticks.index(tickval)]
        xtickValues.append(xtickValue)
    
    return xtickValues,xtickLabels

def setXandYTicks(plottingDf,fg,subPlotType,kwargs):
    #Get GFI xtick values
    xtickValues,xtickLabels = returnTicks([-1000,1000,10000,100000])
    if subPlotType == 'kde':
        #Get count ytick values from histograms
        g = sns.FacetGrid(plottingDf,sharey=False,legend_out=True,**kwargs)
        g.map(sns.distplot,'GFI',bins=256,kde=False)
        ylabels = []
        for axis in plt.gcf().axes:
            ylabels.append(list(map(int,axis.get_yticks().tolist())))
        plt.clf()
    #Add appropriate xtick values (also ytick values if kde) for each axis in figure
    for axis,i in zip(fg.fig.get_axes(),range(len(fg.fig.get_axes()))):
        axis.set_xticks(xtickValues)
        axis.set_xticklabels(xtickLabels)
        if subPlotType == 'kde':
            axis.set_yticklabels(ylabels[i])
