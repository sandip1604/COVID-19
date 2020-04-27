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

path = r'D:\study\Semester 2\Artificial Intelligence\COVID-19\covid_19_clean_complete.csv' # use your path
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

path = r'D:\study\Semester 2\Artificial Intelligence\COVID-19\WPP2019_TotalPopulationBySex.csv'
pop_df = pd.read_csv(path, index_col=None, header=0)
pop_df = pop_df[pop_df['Time'] == 2020]
pop_df = pop_df[pop_df['Variant'] == 'Medium']
pop_df = pop_df[['Location','PopTotal','PopDensity']]
print(pop_df)
df2 = pd.merge(df, pop_df, how='inner', left_on='Country/Region', right_on='Location')

##### other insight on the countires like Gdp and urban population   ########

path = r'D:\study\Semester 2\Artificial Intelligence\COVID-19\Country_Data.csv'
cont_df = pd.read_csv(path, index_col=None, header=0)
cont_df = cont_df[['Location','Urban population%','GDP_per_capita_USD','Current_health_expenditure_per_capita_USD']]
cont_df = cont_df.dropna()
df3 = pd.merge(df2, cont_df, how='inner', left_on='Country/Region', right_on='Location')

df_ex1 = df[['Country/Region']]
df_ex2 = cont_df[['Location']]
df_ex1.columns = ['Location']
a = set(df_ex1.Location)
b = set(df_ex2.Location)
print(a-b)

#train_frame_grouped = train_frame.groupby(['Country])
#for index_rows,row in frame.iterrows():
#    if row['Country/Region'] == 'US' or row['Country_Region'] == 'US':
#        frame.drop(frame.index[index_rows])
#        