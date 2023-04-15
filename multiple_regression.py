import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

data = pd.read_csv('data.csv')

data = data.drop(['Car', 'Model'], axis=1)
data = data.dropna()

X = np.array(data.drop('CO2', axis=1))
X = sm.add_constant(X)
y = np.array(data['CO2'])
model = sm.OLS(y,X).fit()

print(model.summary())

new_data = pd.DataFrame({'Const': [1, 1, 1],'Volume': [900, 1400, 1100], 'Weight': [2300, 1800, 2100]})

prediction = model.predict(new_data)
print("Prediction: ")
print(prediction)