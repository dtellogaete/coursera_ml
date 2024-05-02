# -*- coding: utf-8 -*-
"""
Created on Wed May  1 20:13:30 2024

@author: Daniel
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Sample Submission
df = pd.read_csv('sample_submission.csv')

# Training Set
training_set = pd.read_csv('train.csv')   

# Variables
learning_rate = 0.1
iterations = 100

X_value = training_set['LotArea'].values.reshape(-1, 1)
X_value = np.insert(X_value, 0, 1, axis=1)
y_value = training_set['SalePrice'].values.reshape(-1, 1)

# Escalado de las variables

st_x = StandardScaler()
X_value = st_x.fit_transform(X_value)

st_y = StandardScaler()
y_value = st_y.fit_transform(y_value)

theta = np.zeros((X_value.shape[1], 1))

m = X_value.shape[0]

const = learning_rate * (1 / m)

# Iteraciones
for i in range(iterations):
    h = np.dot(X_value, theta)
    
    # Actualización de theta
    theta[0] = theta[0] - const * np.sum(h - y_value)
    theta[1:] = theta[1:] - const * np.dot(X_value[:, 1:].T, h - y_value)

    if i % 10 == 0:
        cost = np.sum((h - y_value) ** 2) / (2 * m)
        print(f"Iteración {i}: Costo = {cost}")

print("Theta final:", theta)

# Test
test_set = pd.read_csv('test.csv')

X_test_value =  test_set['LotArea'].values.reshape(-1, 1)
X_test_value = np.insert(X_test_value, 0, 1, axis=1)
X_test_value = st_x.fit_transform(X_test_value)

sample_id = test_set['Id'].values.reshape(-1,1)
sample_saleprice = np.dot(X_test_value, theta)

# Desescalar las predicciones en el conjunto de prueba
sample_saleprice_descaled = st_y.inverse_transform(sample_saleprice.reshape(-1, 1))

# Crear DataFrame con las predicciones desescaladas y el Id
sample_id = test_set['Id'].values.reshape(-1, 1)
sample_df = pd.DataFrame({'Id': sample_id.flatten(), 'SalePrice': sample_saleprice_descaled.flatten()})




sample_df.to_csv('sample.csv', index=False)



