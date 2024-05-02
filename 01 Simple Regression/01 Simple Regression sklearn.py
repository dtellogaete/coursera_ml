# -*- coding: utf-8 -*-
"""
Created on Wed May  1 18:41:49 2024

@author: Daniel

"""

import pandas as pd
import numpy as np
# Preprocesado y modelado
# ==============================================================================
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Datos
# ==============================================================================
equipos = ["Texas","Boston","Detroit","Kansas","St.","New_S.","New_Y.",
           "Milwaukee","Colorado","Houston","Baltimore","Los_An.","Chicago",
           "Cincinnati","Los_P.","Philadelphia","Chicago","Cleveland","Arizona",
           "Toronto","Minnesota","Florida","Pittsburgh","Oakland","Tampa",
           "Atlanta","Washington","San.F","San.I","Seattle"]
bateos = [5659,  5710, 5563, 5672, 5532, 5600, 5518, 5447, 5544, 5598,
          5585, 5436, 5549, 5612, 5513, 5579, 5502, 5509, 5421, 5559,
          5487, 5508, 5421, 5452, 5436, 5528, 5441, 5486, 5417, 5421]

runs = [855, 875, 787, 730, 762, 718, 867, 721, 735, 615, 708, 644, 654, 735,
        667, 713, 654, 704, 731, 743, 619, 625, 610, 645, 707, 641, 624, 570,
        593, 556]

datos = pd.DataFrame({'equipos': equipos, 'bateos': bateos, 'runs': runs})
datos.head(3)

# División de los datos en train y test
# ==============================================================================
X = datos[['bateos']]
y = datos['runs']

X_train, X_test, y_train, y_test = train_test_split(
                                        X.values.reshape(-1,1),
                                        y.values.reshape(-1,1),
                                        train_size   = 0.8,
                                        random_state = 1234,
                                        shuffle      = True
                                    )

# Creación del modelo
# ==============================================================================
modelo = LinearRegression()
modelo.fit(X = X_train.reshape(-1, 1), y = y_train)

# Información del modelo
# ==============================================================================
print("Intercept:", modelo.intercept_)
print("Coeficiente:", list(zip(X.columns, modelo.coef_.flatten(), )))
print("Coeficiente de determinación R^2:", modelo.score(X, y))


# Datos 
# ==============================================================================

# Training set 
training_set = pd.read_csv('train.csv') 
X_train = training_set['LotArea'].values.reshape(-1, 1)
y_train = training_set['SalePrice'].values.reshape(-1, 1) 

# Escalado de variables 

st_x = StandardScaler()
X_train = st_x.fit_transform(X_train)

st_y = StandardScaler()
y_train = st_y.fit_transform(y_train)
  

# Test set
test_set = pd.read_csv('test.csv')


# Creación del modelo
# ==============================================================================
modelo = LinearRegression()
modelo.fit(X = X_train.reshape(-1, 1), y = y_train)

# Información del modelo
# ==============================================================================
print("Intercept:", modelo.intercept_)
print("Coeficiente:",  modelo.coef_.flatten())
print("Coeficiente de determinación R^2:", modelo.score(X, y))


theta = np.array([modelo.intercept_, modelo.coef_.flatten()])

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


sample_df.to_csv('sample_sklearn.csv', index=False)


