"""
	Author : (Sathish.V)

	Description : 
		In this project the focus will be on predicting the area of square,
		given the feature (i.e. side of the square).
		Everything in this project was built from scratch.

	Title :
		Predicting the Area of the Square - Linear Regression
	
"""

# importing necessary modules

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from _regression import *

# preparing dataset

feature_array = np.array([x for x in range(1, 10 ** 5 + 1)])

actual_value_array = np.array([x * x for x in range(1, 10 ** 5 + 1)])

x_train, x_test, y_train, y_test = train_test_split(feature_array, actual_value_array, test_size=0.3)

# building the model


# initial-values of slope, bias, learning-rate(alpha)

slope = 0
bias = 0
alpha = 9e-8

# number of updation of slope and bias

epochs = 1000

# No need for feature scaling because of single feature

# storing the value of cost at each step for visualization

cost_list = []
slope_list = []


for _ in range(epochs):

    cost = __compute_cost__(slope, x_train, y_train, bias)

    cost_list.append(cost)
    slope_list.append(slope)

    temp_slope = slope - alpha * __compute_dw__(slope, feature_array, actual_value_array, bias)

    temp_bias = bias - alpha * __compute_db__(slope, feature_array, actual_value_array, bias)

    # updation

    slope, bias = temp_slope, temp_bias


# printing the mean absolute error value

print("mean absolute error : ",__error_factor__(__prediction__(x_test, slope, bias), y_test))

# visualizing the gradient-descent

plt.title("Linear Regression")
plt.xlabel("slope")
plt.ylabel("cost")
plt.plot(slope_list, cost_list)
plt.show()

