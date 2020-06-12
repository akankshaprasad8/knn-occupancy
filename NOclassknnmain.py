# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 16:24:44 2020

@author: Akanksha
"""

import numpy as np
from collections import Counter
import json
import matplotlib.pyplot as plt
import pandas as pd
#from sklearn.pipeline import Pipeline
from collections import Counter
# Importing the dataset
dataset = pd.read_csv('kn1234.csv')
X = dataset.iloc[:, 0:64].values
y = dataset.iloc[:, 64].values

#new jason to data

jsondata = '{"number": 1.573937639}'
array=r'{"AMG8833": {"0_16":[24.50,25.00,24.50,24.25,24.75,24.50,24.75,25.00,25.25,24.75,24.75,24.75,25.50,25.00,25.25,25.25],"16_32":[25.00,25.00,25.25,25.50,25.25,25.25,25.25,26.00,25.50,25.00,25.00,25.50,25.50,25.25,25.75,25.75],"32_48":[25.50,25.50,25.00,25.25,25.00,26.00,26.00,26.25,26.25,25.25,25.00,25.50,25.25,25.75,25.75,26.75],"48_64":[26.25,26.00,26.00,25.75,26.00,25.75,26.25,27.25,27.75,27.00,26.00,26.50,25.75,27.00,27.00,28.00]}}'
list=[]
x = json.loads(array)
for i in x["AMG8833"]["0_16"]:
  list.append(i)
for i in x["AMG8833"]["16_32"]:
  list.append(i)  
for i in x["AMG8833"]["32_48"]:
  list.append(i)  
for i in x["AMG8833"]["48_64"]:
  list.append(i)  

df = pd.DataFrame([list]) 
X_test1 = df.iloc[:, 0:64].values
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.0, random_state = 0)
X_train=X
y_train=y
def euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))

def predict(X):
        y_pred = [_predict(x) for x in X]
        return np.array(y_pred)
def _predict(x):
        # Compute distances between x and all examples in the training set
        distances = [euclidean_distance(x, x_train) for x_train in X_train]
        # Sort by distance and return indices of the first k neighbors
        k_idx = np.argsort(distances)[:5]
        # Extract the labels of the k nearest neighbor training samples
        k_neighbor_labels = [y_train[i] for i in k_idx]  
        # return the most common class label
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]

predction = predict(X_test1)