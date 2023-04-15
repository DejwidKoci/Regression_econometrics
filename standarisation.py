import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

data = pd.read_csv('data.csv')

data = data.drop(['Car', 'Model'], axis=1)
data = data.dropna()

X = np.array(data.drop('CO2', axis=1))
scaled_X = scale.fit_transform(X)
scaled_X = sm.add_constant(scaled_X)
y = np.array(data['CO2'])
model = sm.OLS(y,scaled_X).fit()
#print(scaled_X)
#print(y)

print(model.summary())

scaled = scale.transform([[900,2300],[1400,1800],[1100,2100]])
scaled = sm.add_constant(scaled)

new_data = pd.DataFrame(scaled)
print(model)
prediction = model.predict(new_data)
print("Prediction: ")
print(prediction)