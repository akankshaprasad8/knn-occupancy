# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 12:13:34 2020

@author: Akanksha
"""
# K-Nearest Neighbors (K-NN)
# Importing the libraries
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from sklearn.pipeline import Pipeline
from collections import Counter
# Importing the dataset
dataset = pd.read_csv('kn1234.csv')
X = dataset.iloc[:, 0:64].values
y = dataset.iloc[:, 64].values

#df=pd.read_csv('eveningdata.csv')

#new jason to data
from decimal import Decimal
jsondata = '{"number": 1.573937639}'
array=r'{"AMG8833": {"0_16":[],"16_32":[25.00,25.00,25.25,25.50,25.25,25.25,25.25,26.00,25.50,25.00,25.00,25.50,25.50,25.25,25.75,25.75],"32_48":[25.50,25.50,25.00,25.25,25.00,26.00,26.00,26.25,26.25,25.25,25.00,25.50,25.25,25.75,25.75,26.75],"48_64":[26.25,26.00,26.00,25.75,26.00,25.75,26.25,27.25,27.75,27.00,26.00,26.50,25.75,27.00,27.00,28.00]}}'
list=[]
a=[]
b=[]
x = json.loads(array)
for i in x["AMG8833"]["0_16"]:
  list.append(i)
for i in x["AMG8833"]["16_32"]:
  list.append(i)
  b.append(i)  
for i in x["AMG8833"]["32_48"]:
  list.append(i)  
for i in x["AMG8833"]["48_64"]:
  list.append(i)  

x=[a,b]
d=pd.DataFrame(x)
df = pd.DataFrame([list]) 
X_test1 = df.iloc[:, 0:64].values
# Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.0, random_state = 0)


# Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test1 = sc.transform(X_test1)
# -*- coding: utf-8 -*-


import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))



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
        #print(most_common[0][0])
        return most_common[0][0]

k = 5
clf = KNN(k=k)
clf.fit(X_train, y_train)
prediction = clf.predict(X_test1)
#a=prediction.tolist()
#print(a)

#print("dtype:", prediction.dtype)

#from sklearn.neighbors import NearestCentroid
#clf = NearestCentroid()
#clf.fit(X_train, y_train)

#y_predbynearestneighbour=clf.predict(X_test1)
# Fitting K-NN to the Training set
#from sklearn.neighbors import KNeighborsClassifier
#classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
#classifier.fit(X_train, y_train)
#y_pred_ans=[]
# Predicting the Test set results
#y_pred = classifier.predict(X_test1)

# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix

#cm = confusion_matrix(y_predbynearestneighbour, y_pred)
array=r'{"AMG8833": {"0_16":[24.50,25.00,24.50,24.25,24.75,24.50,24.75,25.00,25.25,24.75,24.75,24.75,25.50,25.00,25.25,25.25],"16_32":[25.00,25.00,25.25,25.50,25.25,25.25,25.25,26.00,25.50,25.00,25.00,25.50,25.50,25.25,25.75,25.75],"32_48":[25.50,25.50,25.00,25.25,25.00,26.00,26.00,26.25,26.25,25.25,25.00,25.50,25.25,25.75,25.75,26.75],"48_64":[26.25,26.00,26.00,25.75,26.00,25.75,26.25,27.25,27.75,27.00,26.00,26.50,25.75,27.00,27.00,28.00]}}'
a=[]
b=[]
c=[]
d=[]
e=[]
f=[]
g=[]
h=[]
x = json.loads(array)
j=0
for i in x["AMG8833"]["0_16"]:
  j=j+1  
  if(j<8):
    a.append(i)
  else:
    b.append(i) 
j=0
for i in x["AMG8833"]["16_32"]:
  j=j+1
  if(j<8):
    c.append(i)
  else:
    d.append(i) 
j=0    
for i in x["AMG8833"]["32_48"]:
  j=j+1
  if(j<8):
    e.append(i)
  else:
    f.append(i) 
j=0    
for i in x["AMG8833"]["48_64"]:
  j=j+1
  if(j<8):
    g.append(i)
  else:
    h.append(i) 
    
    
a=[]
b=[]
c=[]
d=[]
e=[]
f=[]
g=[]
h=[]
#x=[a,b,c,d,e,f,g,h]
j=0
y=[23.75,23.75,24.00,24.00,24.50,24.25,24.25,23.50,23.75,23.50,24.00,24.00,24.75,24.50,23.50,22.25,23.75,23.00,23.25,25.25,25.50,24.25,23.00,23.00,23.00,23.50,23.25,25.25,24.75,24.00,23.75,22.50,22.75,22.75,23.50,24.00,24.25,23.50,23.25,23.00,22.50,22.25,22.75,23.25,22.75,23.00,23.00,22.00,21.50,22.25,22.50,22.50,22.75,23.00,22.50,21.25,19.50,20.50,20.75,21.50,21.75,21.75,21.75,19.50]
#y=[26.50,26.50,26.50,27.00,27.25,27.25,26.75,26.00,25.75,26.25,26.75,27.00,26.75,26.50,26.75,25.50,26.00,26.00,26.50,26.75,27.25,27.00,27.25,26.50,26.25,26.25,27.00,26.75,27.25,26.75,27.00,26.00,25.50,26.25,25.75,26.25,27.00,27.25,26.50,26.00,25.00,25.50,26.25,26.50,26.50,26.25,27.00,25.50,24.25,24.50,26.00,26.00,27.25,26.00,26.00,25.00,21.75,23.50,24.25,25.00,24.75,25.00,24.75,23.25]
#y=[25.75,26.50,27.00,27.50,26.75,27.00,27.00,26.25,26.50,26.50,27.00,27.00,27.25,27.00,27.25,26.00,26.25,26.50,26.50,26.75,27.25,27.25,27.00,26.75,25.50,27.25,26.50,27.25,27.25,27.50,27.00,26.25,25.75,27.00,26.75,26.75,27.50,27.00,27.25,25.75,24.75,25.75,26.50,26.50,26.75,27.00,26.50,26.00,24.25,25.75,26.00,27.00,26.75,26.75,26.00,25.75,22.25,24.25,24.25,25.00,24.75,24.75,24.75,23.75]
#y=[25.50,26.00,25.25,25.25,26.00,25.50,25.75,24.00,25.25,25.75,25.50,25.75,25.50,26.50,25.50,24.50,26.50,25.50,26.00,25.25,26.50,26.25,26.50,24.25,26.00,25.75,26.00,25.75,25.75,25.75,25.50,24.75,25.75,25.50,25.00,25.00,24.75,25.50,25.00,24.25,25.75,25.25,24.75,24.25,24.25,24.25,24.50,24.00,24.00,24.75,24.50,23.50,23.50,23.75,23.50,23.00,23.25,22.75,22.50,22.50,23.00,22.50,23.00,21.75]
#y=[25.50,26.50,26.50,26.25,27.00,27.25,25.75,24.75,25.75,25.50,26.25,26.75,26.75,26.25,26.00,23.50,25.75,25.50,25.25,26.00,26.50,26.00,24.75,23.50,25.00,25.50,25.75,25.75,25.75,25.00,24.25,24.00,25.00,24.75,24.50,25.00,24.75,25.00,23.75,23.50,24.50,25.25,24.50,24.75,23.75,24.00,23.50,23.50,24.25,25.00,25.25,23.75,24.00,24.00,23.00,22.75,22.50,23.75,23.75,22.50,22.50,22.75,23.00,21.50]
#y=[24.75,25.75,25.75,25.75,25.75,25.75,25.00,23.25,25.00,25.00,25.00,26.00,25.25,25.00,25.00,24.75,24.25,24.50,24.50,25.00,25.00,24.25,25.00,24.25,24.00,24.75,23.75,25.25,24.75,23.50,24.25,23.25,23.50,24.25,23.50,24.75,25.00,23.50,23.50,23.25,23.25,23.50,22.50,23.50,24.25,22.50,23.00,23.25,21.75,22.50,22.50,23.50,23.00,23.00,23.25,23.00,20.50,20.75,21.50,21.50,22.00,22.00,22.25,20.00]
#y=[21.25,21.25,18.75,19.5,20.75,19.5,18.25,18.25,21,20.25,19,19.75,20.5,20.75,18.75,18.5,20,19.5,19,19,19.75,19.75,19.25,19.25,19.5,19,19.25,18.75,19,19,18.75,19.5,19.5,19,19,20.25,21.75,20,19.25,19.25,19.75,20.75,20.75,22,22,20.75,19.25,19.5,20.5,21.75,20.5,22,21.75,21.25,19,19.25,20.5,20,19,21,22.25,21.25,20,19.75]
for i in y:
    j=j+1
    if(j<8):
        a.append(i)
    elif(j<16):
        b.append(i)
    elif(j<24):
        c.append(i)
    elif(j<32):
        d.append(i)
    elif(j<40):
        e.append(i)
    elif(j<48):
        f.append(i)
    elif(j<56):
        g.append(i)
    elif(j<64):
        h.append(i)    
x=[a,b,c,d,e,f,g,h]        
y=pd.DataFrame(x)
import seaborn as sns
plt.figure(figsize=(2,2))
#cor=c.corr()
sns.heatmap(y)
#sns.heatmap(cor,annot=True,cmap=plt.cm.Reds)
plt.show()
#from numpy import asarray
#image2 = Image.fromarray(y)
#print(type(image2))
#A=y.to_numpy()
#from PIL import Image
#import numpy as np
#w,h=512,512
#t=(h,w,3)
#A=np.zeros(t,dtype=np.uint8)
#i=Image.fromarray(A,"RGB")
#i.show()
#from PIL import Image
#import numpy as np
#img = Image.fromarray(A, 'RGB')

#newsize = (300, 300) 
#img=img.resize(newsize) 
#img.save('my.png')
#img.show()
print("Sensor should have clear path to calibrate against environment")
#graph = plt.imshow(np.reshape(np.repeat(0,64),(8,8)),cmap=plt.cm.hot,interpolation='lanczos')
graph = plt.imshow(y,cmap=plt.cm.hot,interpolation='lanczos')
plt.colorbar()
plt.clim(1,8) # can set these limits to desired range or min/max of current sensor reading
plt.draw()
norm_pix=y
cal_vec=[]
cal_size=10
k=7

cal_vec=y
for xx in range(0,len(norm_pix)):
        print(xx)
        cal_vec[xx]=norm_pix[xx]
        if kk==cal_size:
                cal_vec[xx] = cal_vec[xx]/cal_size
# Visualising the Training set results
