import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import sklearn.metrics

# load csv file
loaded_csv = pd.read_csv("E:\Developing\MachineLearning\Regression\dataset.csv")

# get data from loaded csv file
x_train_data = loaded_csv['X']
y_train_data = loaded_csv['Y']
x_test_data_arr = loaded_csv['X_TEST']

# reshape all data
x_data = x_train_data.values.reshape(-1, 1)
y_data = y_train_data.values.reshape(-1, 1)
x_test_data = x_test_data_arr.values.reshape(-1, 1)

# Create Linear Regression model
model = LinearRegression()
model.fit(x_data, y_data)
y_pred = model.predict(x_test_data)

# Calculate SMSE, MSE, MAE
print("Coefficient : %.2f" % model.coef_)
print("intercept : %.2f" % model.intercept_)
print("MSE = %.2f " % sklearn.metrics.mean_squared_error(y_data, y_pred))
print("MAE = %.2f " % sklearn.metrics.mean_absolute_error(y_data, y_pred))
print("RMSE = %.2f " % np.sqrt(sklearn.metrics.mean_squared_error(y_data, y_pred)))

# plot the data in matplot lib view
plt.scatter(x_data, y_data, color="red")
plt.plot(x_test_data, y_pred, color="blue")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("X->Y")
plt.show()
print (y_pred)





