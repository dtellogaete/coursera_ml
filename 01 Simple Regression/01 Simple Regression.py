# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 20:48:06 2024

@author: Daniel
"""
import pandas as pd 
import numpy as np


# Sample Submission
df = pd.read_csv('sample_submission.csv')

# Training Set
training_set = pd.read_csv('train.csv')   

# Variables
learning_rate = 0.1
iterations = 100000

X_value = training_set['LotArea'].values.reshape(-1, 1)
X_value = np.insert(X_value, 0, 1, axis = 1)
y_value = training_set['SalePrice'].values.reshape(-1, 1)    
    
theta = np.array([0,0])

m = X_value.shape[0]

const = learning_rate*(1/m)

# Escalado de las variables
from sklearn.preprocessing import StandardScaler
st_x = StandardScaler()
X_value = st_x.fit_transform(X_value)

st_y = StandardScaler()
y_value = st_y.fit_transform(y_value)

# Iteraciones
for i in range(0, iterations):
    h = np.dot(X_value, theta)
    
    theta[0] = theta[0]-const* np.sum(h-y_value)
    theta[1] = theta[1] - const * np.sum(h - np.dot(y_value.T, X_value[:, 1]))
    print(const * np.sum(h - np.dot(y_value.T, X_value[:, 1])))
    print('iteracion' ,i, theta )
       
# Test Set
test_set = pd.read_csv('test.csv')

print(theta)




