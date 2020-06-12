# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 20:22:23 2020

@author: Akanksha
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 12:13:34 2020

@author: Akanksha
"""

# K-Nearest Neighbors (K-NN)

# Importing the libraries
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sns
import json
#from decimal import Decimal
#from sklearn.pipeline import Pipeline
from collections import Counter
# Importing the dataset

import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))


def testdata():
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
     return X_test1

class KNN:

    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_idx = np.argsort(distances)[:self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_neighbor_labels = [self.y_train[i] for i in k_idx]  
        # return the most common class label
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]

  
def main():
  dataset = pd.read_csv('kn1234.csv')
  X = dataset.iloc[:, 0:64].values
  y = dataset.iloc[:, 64].values
  
  X_test1=testdata()
  k = 5
  clf = KNN(k=k)
  clf.fit(X, y)
  pr= clf.predict(X_test1)  
  print(pr)
if __name__ == "__main__":
  main()
  

