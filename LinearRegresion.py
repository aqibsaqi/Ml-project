#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

# Load the diabetes dataset
diabetes = datasets.load_diabetes()

# Use only one feature for simplicity
diabetes_x = diabetes.data[:, np.newaxis, 2]

# Split the data into training and testing sets
diabetes_x_train = diabetes_x[:-30]
diabetes_x_test = diabetes_x[-30:]

diabetes_y_train = diabetes.target[:-30]
diabetes_y_test = diabetes.target[-30:]

# Create a Linear Regression model
model = linear_model.LinearRegression()  # Fixed the typo here

# Train the model
model.fit(diabetes_x_train, diabetes_y_train)

# Make predictions
diabetes_y_predict = model.predict(diabetes_x_test)

# Evaluate the model
print("Mean squared error:", mean_squared_error(diabetes_y_test, diabetes_y_predict))

# Plot outputs
import matplotlib.pyplot as plt

plt.scatter(diabetes_x_test, diabetes_y_test, color="black")
plt.plot(diabetes_x_test, diabetes_y_predict, color="blue", linewidth=3)

plt.xlabel("Feature")
plt.ylabel("Target")
plt.title("Linear Regression on Diabetes Dataset")
plt.show()


# In[ ]:




