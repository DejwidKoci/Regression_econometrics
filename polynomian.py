import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# data
x = np.array([1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22])
y = np.array([100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100])


# creation of matrices of independent variables
X = np.column_stack((x, x**2, x**3))

# adding constant
X = sm.add_constant(X)

# model fitting using the method of least squares
model = sm.OLS(y, X).fit()

# show summary
print(model.summary())

# creation of a grid of x-points on the x-axis
x_grid = np.linspace(x.min(), x.max(), 100)

# obliczenie przewidywanych wartości y dla siatki punktów x
X_grid = np.column_stack((x_grid, x_grid**2, x_grid**3))
X_grid = sm.add_constant(X_grid)
y_pred = model.predict(X_grid)

# narysowanie punktów danych i krzywej regresji
plt.scatter(x, y)
plt.plot(x_grid, y_pred)
plt.show()


