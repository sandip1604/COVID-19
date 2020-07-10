# -*- coding: utf-8 -*-
"""
Created on Sat May 16 13:35:12 2020

@author: Sandip

"""
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree
import datetime

import warnings
warnings.filterwarnings("ignore")

def display(df):
    print(df)

path = r'C:\Users\Sandip\Documents\Covid_19\covid_19_clean_complete.csv' # use your path
df = pd.read_csv(path, index_col=None, header=0)
df['Date'] = pd.to_datetime(df['Date'])
number_of_countries = 0
df = df.drop(['Province/State'], axis = 1)
lat_df = df[['Location','Lat','Long']]
lat_df = lat_df.drop_duplicates()
df = df.drop(['Long','Lat'], axis = 1)
df = df.groupby(by=['Location','Date'], as_index = False).sum()
df = df.drop_duplicates()
#-----------------------------------------------------------------------------

path = r'C:\Users\Sandip\Documents\Covid_19\WPP2019_TotalPopulationBySex.csv'
pop_df = pd.read_csv(path, index_col=None, header=0)
pop_df = pop_df[pop_df['Time'] == 2020]
pop_df = pop_df[pop_df['Variant'] == 'Medium']
pop_df = pop_df[['Location','PopTotal','PopDensity']]
print(pop_df)
df2 = pd.merge(df, pop_df, how='inner', left_on='Location', right_on='Location')

##### other insight on the countires like Gdp and urban population   ########

path = r'C:\Users\Sandip\Documents\Covid_19\Country_Data.csv'
cont_df = pd.read_csv(path, index_col=None, header=0)
cont_df = cont_df[['Location','Urban population%','GDP_per_capita_USD','Current_health_expenditure_per_capita_USD']]
cont_df = cont_df.dropna()
df3 = pd.merge(df2, cont_df, how='inner', left_on='Location', right_on='Location')
df3['Days_10'] = 0

#################################################################################################################################
grouped_by_country_province = df3.groupby(['Location'])

column_names = df3.keys()

days_df = pd.DataFrame(columns = column_names)
for name, group in grouped_by_country_province:
    initial_day = datetime.datetime.now()
    for index, row in group.iterrows():
        if row['Confirmed']<10:
            initial_day = row['Date']
        if row['Confirmed']>= 10:
        
            delta = (row['Date']-initial_day).days
            group.set_value(index,'Days_10', delta)
    days_df = days_df.append(group)      
days_df = days_df.drop(['Date'], axis = 1)
days_df = days_df[days_df['Days_10']!=0]

#####################################################################################################


######    making testing nd training sets from the data frame   ####################

df_ml = pd.get_dummies(days_df, columns = ['Location'])

##### Data visualisation for few of the countries  ########################


US_df = df_ml[df_ml['Location_US']==1]
X_v = US_df['Days_10'].values.tolist()
Y_v = US_df['Confirmed'].values.tolist()
Z = US_df['Deaths'].values.tolist()
P = US_df['Recovered'].values.tolist()
#s = sns.lineplot(x = X_v, y = Y_v)
#s = sns.lineplot(x = X_v, y = Z)
plt.plot(X_v, Y_v,X_v,Z,X_v,P)
plt.legend(['Confirmed', 'Deaths', 'Recovered'] ,ncol=2, loc='upper left')
plt.title('United States stats')
plt.xlabel('Days')
plt.ylabel('Number of people')
#### training and testing   ########################

### shuffling the data before spliting 
shuffle_df_ml = df_ml.reindex(np.random.permutation(df_ml.index))
shuffle_df_ml_X = shuffle_df_ml.drop(['Confirmed','Deaths','Recovered'], axis = 1)
shuffle_df_ml_Y = shuffle_df_ml[['Confirmed','Deaths','Recovered']]

#------------- arrays from dataframes----------------------#
X = shuffle_df_ml_X.values
Y = shuffle_df_ml_Y.values
#------------------ spliting test and train with shuffling again to be dubble sure -----------------------

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42, shuffle = True)

#----------------- scaling features for centering and stuff ------------------------------#
scaler = StandardScaler()                           # standard scaler
scaler.fit(X_train)                                 # fiting the scaling model with training data
X_train = scaler.transform(X_train)                 # transform training data
X_test = scaler.transform(X_test)                   #transform testing data
clf = tree.DecisionTreeRegressor()                  #decision tree regressor as ML model
clf = clf.fit(X_train,y_train)                      #training the decsion tree
final = clf.predict(X_test)                         #predicting the data

#---------------- evaluating model using differnet metrics ------------------------------#

print("R2 score of the test set: " + str(metrics.r2_score(y_test, final)))              
print("explained varinace score of the test set: "+ str(metrics.explained_variance_score(y_test,final, multioutput = 'uniform_average')))

#--------------- model just for confirmed cases  ------------------------------------#

                     #predicting the data

#---------------- evaluating model using differnet metrics ------------------------------#



#------------------- trying diferent models-------------------------------------------# 

#------------------- SVM regressor --------------------------------------------
X_try = np.array([(331002.647,	36.185,	82.26,	62794.59,	10246.14,	148,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,  0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0), (331002.647,	36.185,	82.26,	62794.59,	10246.14,	149,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,  0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,)])
X_try_fit = scaler.transform(X_try)
Y_pred = clf.predict(X_try_fit)
