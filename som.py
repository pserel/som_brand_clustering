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

# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, 
              y = 10, 
              input_len = 15,
              sigma = 1.0,
              learning_rate = 0.5)
som.random_weights_init(data = X)
som.train_random(data = X,
                 num_iteration = 100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
from random import random
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    jitter_x = random() * 0.4
    jitter_y = random() * 0.4
    offset_x = 0.1
    if y[i] == 1:
        offset_x = .5
    plot(w[0] + jitter_x + offset_x,
         w[1] + jitter_y + 0.25,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 2)
show()

# Finding the frauds (after visual inspection)
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(7, 4)], mappings[(8, 4)]), axis = 0)
frauds = sc.inverse_transform(frauds)



# Finding the frauds (based on the distances)
distances = som.distance_map().T
mappings = som.win_map(X)
frauds2 = []
 
for yy in range(len(distances)):
  for xx in range(len(distances[yy])):
    if distances[yy, xx] >= 0.98: # 1.0 is white
      frauds2.extend(mappings[(xx, yy)])
    
frauds2 = sc.inverse_transform(frauds2)


