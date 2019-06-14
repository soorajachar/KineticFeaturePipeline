#!/usr/bin/env python3 
import json,pickle,math,matplotlib,sys,os,string
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

folderName = '20190404-CD25MutantTimeSeries_OT1_Timeseries_2'
dataTypeList = ['cyt']
idx = pd.IndexSlice
mi_df = pickle.load(open('../kineticFeatureOutput/mutualInfoMetric-'+folderName+'.pkl','rb')).loc[idx[dataTypeList],:]
cv_df = pickle.load(open('../kineticFeatureOutput/cvMetric-'+folderName+'.pkl','rb')).loc[idx[dataTypeList],:]
combined_df = pd.concat([mi_df,cv_df],axis=1)
plottingDf = combined_df.reset_index()

newtslist = []
sns.set_palette(sns.color_palette("magma", len(plottingDf['TimeSlice'])))
ax = sns.relplot(x='Quality',y='Quantity',hue='TimeSlice',data=plottingDf)#,facet_kws={'legend_out':False})
#ax._legend = plt.legend(ncol=4,bbox_to_anchor=(1,1),frameon=False)
plt.show()
#plt.savefig('wat2.png',bbox_inches='tight')
"""
tips = sns.load_dataset("tips")
ax = sns.relplot(x="total_bill", y="tip", hue='time',kind='line',data=tips,facet_kws={'legend_out':False})
ax._legend = plt.legend(ncol=3)
plt.show()
"""
