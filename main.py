# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 07:55:46 2020

@author: SERELPA1
"""

# Self Organizing Map

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# pip install varclushi
# from varclushi import VarClusHi

# Importing the dataset
dataset = pd.read_csv('pharma_products_ts.csv')

# Process the dataset
# list(dataset.columns.values)
dataset['product_id'] = dataset['country_new'] + '_' + dataset['product_new']

dataset_yoy_vol_sum = dataset.groupby(['product_id', 'year'], as_index = False)[['volume']].agg(sum)
dataset_yoy_vol_sum['volume_lag1'] = dataset_yoy_vol_sum.groupby(['product_id'], as_index = False)[['volume']].shift(1)
dataset_yoy_vol_sum['volume_yoy'] = dataset_yoy_vol_sum['volume'] / dataset_yoy_vol_sum['volume_lag1'] - 1

dataset_mom_vol_sum = dataset.groupby(['product_id', 'year', 'month'], as_index = False)[['volume']].agg(sum)
dataset_mom_vol_sum['volume_lag1'] = dataset_mom_vol_sum.groupby(['product_id'], as_index = False)[['volume']].shift(1)
dataset_mom_vol_sum['volume_mom'] = dataset_mom_vol_sum['volume'] / dataset_mom_vol_sum['volume_lag1'] - 1

dataset = pd.merge(left = dataset,
                   right = dataset_yoy_vol_sum[['product_id', 'year', 'volume_yoy']],
                   how = 'left',
                   on = ['product_id', 'year'])
dataset = dataset.merge(right = dataset_mom_vol_sum[['product_id', 'year', 'month', 'volume_mom']],
                        how = 'left',
                        on = ['product_id', 'year', 'month'])


cols = ['product_id', 'date', 'months_from_launch_date', 'months_to_loe_date', 'volume_yoy', 'volume_mom']
X = dataset.replace([np.inf, -np.inf], np.nan).dropna(subset = ['volume_yoy', 'volume_mom'])
X = X.loc[:, cols]
X.reset_index(drop = True, inplace = True)
# y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
Xsc = sc.fit_transform(X.iloc[:, 2:6].values)

product_max_date = X.groupby(['product_id'], as_index = False)[['date']].agg(max)
product_max_date.rename({'date': 'max_date'}, axis = 1, inplace = True)
X = pd.merge(left = X,
                   right = product_max_date,
                   how = 'left',
                   on = ['product_id'])

# Training the SOM
from minisom import MiniSom
som = MiniSom(x = 3, 
              y = 2, 
              input_len = 4,
              sigma = 0.2,
              learning_rate = 0.5)

som.random_weights_init(data = Xsc)

som.train_random(data = Xsc,
                 num_iteration = 100)

mappings = som.win_map(Xsc)
distances = som.distance_map().T

segment_keys = list(mappings.keys())
segments = { segment_keys[i-1]: i for i in range(1, 7) }


for i, x in enumerate(Xsc):
    w = som.winner(x)
    X.loc[i, 'segment'] = segments[w]
    

X_ls = X[lambda X: X.date == X.max_date]
y = X_ls.loc[:, 'segment'].values

Xsc_ls = Xsc[X_ls.index, :]


# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
from random import random
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's', 'D', 'X', 'v', 'p']
colors = ['r', 'g', 'y', 'b', 'c', 'm']
for i, x in enumerate(Xsc_ls):
    w = som.winner(x)
    jitter_x = random() * 0.4
    jitter_y = random() * 0.4
    offset_x = 0.1
    if y[i] == 1:
        offset_x = .5
    plot(w[0] + jitter_x + offset_x,
         w[1] + jitter_y + 0.25,
         markers[int(y[i]) - 1],
         markeredgecolor = colors[int(y[i]) - 1],
         markerfacecolor = 'None',
         markersize = 2
         )
show()

# Finding the frauds (after visual inspection)
frauds = np.concatenate((mappings[(0, 0)], 
                                  mappings[(0, 1)],
                                  mappings[(1, 0)],
                                  mappings[(1, 1)]), axis = 0)
frauds = sc.inverse_transform(frauds)



# Finding the frauds (based on the distances)
distances = som.distance_map().T
mappings = som.win_map(Xsc)
frauds2 = []
 
for yy in range(len(distances)):
  for xx in range(len(distances[yy])):
    if distances[yy, xx] >= 0.98: # 1.0 is white
      frauds2.extend(mappings[(xx, yy)])
    
frauds2 = sc.inverse_transform(frauds2)


