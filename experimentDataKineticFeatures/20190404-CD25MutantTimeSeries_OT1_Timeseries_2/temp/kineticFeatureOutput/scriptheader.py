#!/usr/bin/env python3 
import json,pickle,math,matplotlib,sys,os,string
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

idx = pd.IndexSlice
folderName = '20190404-CD25MutantTimeSeries_OT1_Timeseries_2' 
mi_df = pickle.load(open('fullFeatureDf-'+folderName+'-standardized.pkl','rb'))
