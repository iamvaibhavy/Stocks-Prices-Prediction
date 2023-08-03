# Stocks-Prices-Prediction


import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



dataset = pd.read_csv('NSE-Tata-Limited.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)



print(X_test)


print(X_train)


y_train = y_train.reshape(len(y_train),1)
print(y_train)


y_test = y_test.reshape(len(y_test),1)
print(y_test)



from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)



y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))



from sklearn.metrics import r2_score
r2_score(y_test, y_pred)



print(regressor.predict([[206,224,100,222,221,1567539]]))
