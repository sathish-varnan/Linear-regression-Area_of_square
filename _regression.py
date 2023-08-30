"""
	Author : (Sathish.V)

	Description :
		Important method's definitions to implement the regression
		model.

	Methods:
		1. __compute_unit__(slope, feature, actual_value, bias)
		2. __compute_cost__(slope, feature_array, actual_value_array, bias)
		3. __compute_dw__(slope, feature, actual_value, bias)
		4. __compute_db__(slope, feature, actual_value)
"""

def __compute_unit__(slope, feature, actual_value, bias):
	
	"""
		computes the predicted value of a simple linear regression
		in the formual :=> y = mx + c
	"""
	return (slope * feature + bias) - actual_value


def __compute_cost__(slope, feature_array, actual_value_array, bias):

	"""
		computes the loss of the regression model
	"""

	cost = 0

	iteration = feature_array.shape[0]

	for index in range(iteration):

		cost += __compute_unit__(slope, feature_array[index], actual_value_array[index], bias) ** 2

	cost = cost / (2 * iteration)

	return cost

def __compute_dw__(slope, feature_array, actual_value_array, bias):

	"""
		computes the derivative of the cost with respect to the slope 
	"""

	dw = 0

	iteration = feature_array.shape[0]

	for index in range(iteration):

		dw += __compute_unit__(slope, feature_array[index], actual_value_array[index], bias) * feature_array[index]

	dw = dw / iteration

	return dw

def __compute_db__(slope, feature_array, actual_value_array, bias):
	
	"""
		computes the derivative of the cost with respect to the bias
	"""

	db = 0

	iteration = feature_array.shape[0]

	for index in range(iteration):

		db += __compute_unit__(slope, feature_array[index], actual_value_array[index], bias)

	db = db / iteration

	return db

def __prediction__(test_array, slope, bias):

  """
    computes the prediction and returns the prediction numpy.ndarray
  """

  prediction_array = []

  for index in range(test_array.shape[0]):

    prediction_array.append(slope * test_array[index] + bias)
  
  return np.array(prediction_array)


def __error_factor__(output, expected_output):

	"""
		computes the mean absolute error of the given two samples
	"""

	iteration = output.shape[0]

	error = 0.0

	for index in range(iteration):

		error += (output[index] - expected_output[index]) * (-1 if (output[index] - expected_output[index]) < 0 else 1)

	return error / iteration
