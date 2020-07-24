#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 16:02:49 2020

@author: ellenxiao
"""

import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

file_name = "/Users/autos.csv"
df = pd.read_csv(file_name, encoding="ISO-8859-1")
df.head()

print('Number of Rows: ',df.shape[0])
print('Number of Columns: ', df.shape[1])
print('Features: \n', df.columns.tolist())
print('Number of null value: ', df.isnull().sum().values.sum())
print('Unique values: \n', df.nunique())

# 1.5 IQR rule is (Q1-1.5*(Q3-Q1),Q3+1.5*(Q3-Q1))

lower_bound = df['yearOfRegistration'].quantile(.25)-(df['yearOfRegistration'].quantile(.75)-df['yearOfRegistration'].quantile(.25))*1.5
upper_bound = df['yearOfRegistration'].quantile(.75)+(df['yearOfRegistration'].quantile(.75)-df['yearOfRegistration'].quantile(.25))*1.5
print('Lower bound for yearOfRegistration is',lower_bound)
print('upper bound for yearOfRegistration is',upper_bound)

lower_bound = df['powerPS'].qåçuantile(.25)-(df['powerPS'].quantile(.75)-df['powerPS'].quantile(.25))*1.5
upper_bound = df['powerPS'].quantile(.75)+(df['powerPS'].quantile(.75)-df['powerPS'].quantile(.25))*1.5
print('Lower bound for powerPS is',lower_bound)
print('upper bound for powerPS is',upper_bound)

lower_bound = df['kilometer'].quantile(.25)-(df['kilometer'].quantile(.75)-df['kilometer'].quantile(.25))*1.5
upper_bound = df['kilometer'].quantile(.75)+(df['kilometer'].quantile(.75)-df['kilometer'].quantile(.25))*1.5
print('Lower bound for kilometer is',lower_bound)
print('upper bound for kilometer is',upper_bound)

lower_bound = df['price'].quantile(.25)-(df['price'].quantile(.75)-df['price'].quantile(.25))*1.5
upper_bound = df['price'].quantile(.75)+(df['price'].quantile(.75)-df['price'].quantile(.25))*1.5
print('Lower bound for price is',lower_bound)
print('upper bound for price is',upper_bound)

df_no_outlier = df[
        (df['yearOfRegistration'] <= 2020) 
      & (df['yearOfRegistration'] >= 1985) 
      & (df['powerPS'] >= 10) 
      & (df['powerPS'] <= 500) 
      & (df['kilometer'] >= 5000) 
      & (df['kilometer'] <= 150000)
      & (df['price'] >= 100) 
      & (df['price'] <= 150000)]
print('Data kept for analysis: %d percent of the entire set' % (100*df_no_outlier['name'].count()/df['name'].count()))
df_no_outlier.head()

# drop nrOfPictures columns
df_no_outlier = df_no_outlier.drop(['nrOfPictures'],axis=1)

# since dateCrawled is when the data is first crawled, can be deleted
df_no_outlier = df_no_outlier.drop(['dateCrawled'], axis=1)

# look at seller
print(df_no_outlier.groupby('seller').size(),'\n')
# look at offer type
print(df_no_outlier.groupby('offerType').size(),'\n')
# look at abtest
print(df_no_outlier.groupby('abtest').size(),'\n')
# look at fuel type
print(df_no_outlier.groupby('fuelType').size(),'\n')
# look at not Repaired Damage
print(df_no_outlier.groupby('notRepairedDamage').size(),'\n')

# drop seller and offerType
df_no_outlier = df_no_outlier.drop(['seller'], axis=1)
df_no_outlier = df_no_outlier.drop(['offerType'], axis=1)
df_no_outlier.head()

df_no_outlier.isnull().sum().sort_values(ascending=False)
df_no_outlier = df_no_outlier.dropna(subset=['model'])
df_no_outlier[
    (df_no_outlier['model'].notnull()==True) & (df_no_outlier['vehicleType'].notnull()==False)
]

# create a dataframe that has unique values of model with it's vehicle type
df_model = df_no_outlier.drop_duplicates('model')[['vehicleType','model']]

# convert dataframe to dictionary
dict_model = dict(zip(df_model['model'],df_model['vehicleType']))
dict_model

df_update = df_no_outlier.copy()

for index, row in df_update.iterrows():
    if pd.isna(row['vehicleType']) == True:
        df_update.loc[index,'vehicleType'] = dict_model[row['model']]
        
def convert_gearbox(row):
    if row['gearbox'] == 'manuell':
        row['gearbox'] = 'manual'
    elif row['gearbox'] == 'automatik':
        row['gearbox'] = 'automatic'
    return row['gearbox']

def convert_notRepairedDamage(row):
    if row['notRepairedDamage'] == 'ja':
        row['notRepairedDamage'] = 'yes'
    elif row['notRepairedDamage'] == 'nein':
        row['notRepairedDamage'] = 'no'
    return row['notRepairedDamage']
        
def convert_fuelType(row):    
    if row['fuelType'] == 'diesel':
        row['fuelType'] = 'diesel'
    elif row['fuelType'] == 'benzin':
        row['fuelType'] = 'petrol'
    elif row['fuelType'] =='andere':
        row['fuelType'] = 'other'
    elif row['fuelType'] == 'elektro':
        row['fuelType'] = 'electro'
    
    return row['fuelType']

def convert_vehicleType(row):    
    if row['vehicleType'] == 'kleinwagen':
        row['vehicleType'] = 'small car'
    elif row['vehicleType'] == 'limousine':
        row['vehicleType'] = 'sedan'
    elif row['vehicleType'] =='cabrio':
        row['vehicleType'] = 'convertible car'
    elif row['vehicleType'] == 'kombi':
        row['vehicleType'] = 'station wagon'
    elif row['vehicleType'] == 'andere':
        row['vehicleType'] = 'other'
    
    return row['vehicleType']

df_update['gearbox']= df_update.apply(convert_gearbox, axis=1)
df_update['notRepairedDamage']= df_update.apply(convert_notRepairedDamage, axis=1)
df_update['fuelType']= df_update.apply(convert_fuelType, axis=1)
df_update['vehicleType']= df_update.apply(convert_vehicleType, axis=1)

df_update = df_update.dropna(subset=['vehicleType'])
df_update = df_update.dropna(subset=['notRepairedDamage'])
df_update['fuelType'].fillna(value='not-declared', inplace=True)
df_update['gearbox'].fillna(value='not-declared', inplace=True)
df_update = df_update.drop(['abtest','postalCode','monthOfRegistration'],axis=1)

df_update['daysOnline'] = pd.to_datetime(df_update['lastSeen'])-pd.to_datetime(df_update['dateCreated'])
df_update['daysOnline'] = df_update['daysOnline'].dt.days+1
df_update = df_update.drop(['dateCreated','lastSeen'],axis=1)
df_update.head()

df_update['numberOfYear'] = datetime.now().year - df_update['yearOfRegistration']
#df_update = df_update.drop(['yearOfRegistration'],axis=1)
df_update.head()