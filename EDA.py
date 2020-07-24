#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 16:07:10 2020

@author: ellenxiao
"""

import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

type_count = df_update.groupby(by=['vehicleType'],as_index=False)['name'].count()
gear_count = df_update.groupby(by=['gearbox'],as_index=False)['name'].count()
fuel_count = df_update.groupby(by=['fuelType'],as_index=False)['name'].count()
repair_count = df_update.groupby(by=['notRepairedDamage'],as_index=False)['name'].count()
brand_count = df_update.groupby(by=['brand'],as_index=False)['name'].count()
year_count = df_update.groupby(by=['yearOfRegistration'],as_index=False)['name'].count()

fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Number of cars for different year of registration',
                                    'Number of cars for different gearbox',
                                    'Number of cars for different fuel type',
                                    'Number of cars for damage or not'))
fig.add_trace(
    go.Bar(
        x=year_count['yearOfRegistration'],
        y=year_count['name'],
        marker_color='lightpink'
    ),
    row=1, col=1
)
fig.add_trace(
    go.Bar(
        x=gear_count['gearbox'],
        y=gear_count['name'],
        marker_color='lightcoral'
    ),
    row=1, col=2
)
fig.add_trace(
    go.Bar(
        x=fuel_count['fuelType'],
        y=fuel_count['name'],
        marker_color='moccasin'
    ),
    row=2, col=1
)
fig.add_trace(
    go.Bar(
        x=repair_count['notRepairedDamage'],
        y=repair_count['name'],
        marker_color='paleturquoise'
    ),
    row=2, col=2
)

fig.update_layout(height=800, width=900, title_text="")
fig.show()

# group dataset by brand and vehicleType
model_post = df_update.groupby(by=['brand'],as_index=False)['price','daysOnline'].mean()
type_post = df_update.groupby(by=['vehicleType'],as_index=False)['price','daysOnline'].mean()

# round up daysOnlinåe and price
model_post['daysOnline'] = round(model_post['daysOnline'])
type_post['daysOnline'] = round(type_post['daysOnline'])
model_post['price'] = round(model_post['price'])
type_post['price'] = round(type_post['price'])

fig = make_subplots(
                    rows=1, cols=3,
                    subplot_titles=('Number of cars for different car brands',
                                    'Average days online for different car brands',
                                    'Average price for different car brands'))
fig.add_trace(
    go.Bar(
        x=brand_count['brand'],
        y=brand_count['name'],
        marker_color='khaki'
    ),
    row=1, col=1
)
fig.add_trace(
    go.Bar(
        x=model_post['brand'],
        y=model_post['daysOnline'],
        marker_color='plum'
    ),
    row=1, col=2
)
fig.add_trace(
    go.Bar(
        x=model_post['brand'],
        y=model_post['price'],
        marker_color='lightblue'
    ),
    row=1, col=3
)

fig.update_layout(height=450, width=1100, title_text="")
fig.show()

fig = make_subplots(
                    rows=1, cols=3,
                    subplot_titles=('Number of cars for different car types',
                                    'Average price for different car types',
                                    'Average price online for different car types'))
fig.add_trace(
    go.Bar(
        x=type_count['vehicleType'],
        y=type_count['name'],
        marker_color='moccasin'
    ),
    row=1, col=1
)
fig.add_trace(
    go.Bar(
        x=type_post['vehicleType'],
        y=type_post['daysOnline'],
        marker_color='lightpink'
    ),
    row=1, col=2
)
fig.add_trace(
    go.Bar(
        x=type_post['vehicleType'],
        y=type_post['price'],
        marker_color='lightsteelblue'
    ),
    row=1, col=3
)
fig.update_layout(height=450, width=1100, title_text="")
fig.show()

# group dataset by powerPS
power_brand = df_update.groupby(by=['brand'],as_index=False)['powerPS'].mean()
power_type = df_update.groupby(by=['vehicleType'],as_index=False)['powerPS'].mean()

# round up powerPS
power_brand['powerPS'] = round(power_brand['powerPS'],2)
power_type['powerPS'] = round(power_type['powerPS'],2)

fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Average power for different car brand',
                                    'Average power for different car types'))
fig.add_trace(
    go.Bar(
        x=power_brand['brand'],
        y=power_brand['powerPS'],
        marker_color='wheat'
    ),
    row=1, col=1
)
fig.add_trace(
    go.Bar(
        x=power_type['vehicleType'],
        y=power_type['powerPS'],
        marker_color='thistle'
    ),
    row=1, col=2
)
fig.update_layout(height=500, width=1000, title_text="")
fig.show()

year_price = df_update.groupby(by=['vehicleType','yearOfRegistration'],as_index=False)['price'].mean()
vehicle_type = list(year_price['vehicleType'].unique())
color_list = ['mediumvioletred','deeppink','hotpink','palevioletred','crimson','orchid','violet']
dash_list = ["solid", "dot", "dash", "longdash", "dashdot", "longdashdot", "dot"]

fig = go.Figure()
for i,vType in zip(range(len(vehicle_type)-1),vehicle_type):
    
    fig.add_trace(go.Scatter(x=year_price[year_price['vehicleType']==vType]['yearOfRegistration'], 
                             y=year_price[year_price['vehicleType']==vType]['price'], mode='lines',
        name=vType,
        line=dict(color=color_list[i], width=2, dash=dash_list[i]),
        connectgaps=True,
    ))
fig.update_layout(height=500, width=1000, title_text="Average price of different year of registration of different vehicle types")
fig.show()

func_list = [('mean',np.mean),('25', lambda x: np.percentile(x,.25)),('mediam', np.median),
            ('75', lambda x: np.percentile(x,.75))]
df_pivot_table = df_update.pivot_table(columns=['vehicleType'],index=['yearOfRegistration'],values=['price'],aggfunc=[np.max,np.median])

df_pivot_table.columns = ['_'.join(col) for col in df_pivot_table.columns]
df_pivot_table[df_pivot_table.index > 2013]

df_1516 = df_update[(df_update['yearOfRegistration']==2016) | (df_update['yearOfRegistration']==2015)]
#df_2015_sedan.groupby(['model'])['name'].count()

df_sedan = df_1516.pivot_table(columns=['yearOfRegistration'],values=['name'],index=['brand'],aggfunc=len)
pd.set_option('display.max_rows', df_sedan.shape[0]+1)
df_sedan

df2 = df_update.drop(['name'],axis=1)
df2 = df2.drop(['yearOfRegistration'],axis=1)
df2.head()