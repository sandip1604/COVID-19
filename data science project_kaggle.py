# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 19:36:22 2020

@author: sp3779
"""
import pandas as pd
import seaborn as sns
import numpy as np
import sklearn as sk

def display(df):
    print(df)

path = r'C:\Users\sp3779\Downloads\covid_19_clean_complete.csv' # use your path
df = pd.read_csv(path, index_col=None, header=0)
df['Date'] = pd.to_datetime(df['Date'])
grouped_by_country = df.groupby(['Country/Region'])
#grouped_by_country.apply(display)
number_of_countries = 0
for name,group in grouped_by_country:
    number_of_countries = number_of_countries+1
    print(name," :",len(group))
print(number_of_countries)

#-----------------------------------------------------------------------------

path = r'C:\Users\sp3779\Downloads\WPP2019_TotalPopulationBySex.csv'
pop_df = pd.read_csv(path, index_col=None, header=0)
pop_df = pop_df[pop_df['Time'] == 2020]
pop_df = pop_df[pop_df['Variant'] == 'Medium']
print(pop_df)
  
#train_frame_grouped = train_frame.groupby(['Country])
#for index_rows,row in frame.iterrows():
#    if row['Country/Region'] == 'US' or row['Country_Region'] == 'US':
#        frame.drop(frame.index[index_rows])
#        