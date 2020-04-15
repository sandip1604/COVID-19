# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 17:54:35 2020

@author: sp3779
"""

import pandas as pd
import seaborn as sns
import numpy as np
import sklearn as sk
import glob


path = r'C:\Users\sp3779\Downloads\COVID-19-master\COVID-19-master\csse_covid_19_data\csse_covid_19_daily_reports' # use your path

all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:

    df = pd.read_csv(filename, index_col=None, header=0)
    #print(filename)
    li.append(df)

frame = pd.concat(li, axis = 0, ignore_index = True)
frame = frame.drop(columns = ['Lat','Latitude','Long_','Longitude','FIPS','Combined_Key'])
US_frame = frame[['Active','Admin2','Confirmed','Country/Region','Country_Region','Deaths','Last Update','Last_Update','Province_State','Recovered']]
train_frame = frame[(frame['Country_Region'] != 'US') & (frame['Country/Region'] != 'US')] 
print(train_frame.Country_Region.unique()
#train_frame_grouped = train_frame.groupby(['Country])
#for index_rows,row in frame.iterrows():
#    if row['Country/Region'] == 'US' or row['Country_Region'] == 'US':
#        frame.drop(frame.index[index_rows])
#        