# !/usr/bin/python
# coding=utf-8

# Import necessary libraries
import streamlit as sl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate some sample data
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # Features (independent variable)
y = 2.5 * X + np.random.randn(100, 1) * 2  # Target (dependent variable) with some noise

# Create and train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Model evaluation
print("Model Coefficients:", model.coef_[0][0])
print("Model Intercept:", model.intercept_[0])
print("Mean squared error: %.2f" % mean_squared_error(y, y_pred))
print("Coefficient of determination (R squared score): %.2f" % r2_score(y, y_pred))

# Plot the results
fig=plt.figure()
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, y_pred, color='red', linewidth=2, label='Predicted line')
plt.xlabel('X (Independent Variable)')
plt.ylabel('y (Dependent Variable)')
plt.title('Simple Linear Regression')
plt.legend()
sl.write(fig)

sl.write("Model Coefficients:", model.coef_[0][0])
sl.write("Model Intercept:", model.intercept_[0])
sl.write("Mean squared error: %.2f" % mean_squared_error(y, y_pred))
sl.write("Coefficient of determination (R squared score): %.2f" % r2_score(y, y_pred))