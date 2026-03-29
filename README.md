# BLENDED_LEARNING
# Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import necessary libraries (pandas, numpy, sklearn, matplotlib).

2.Load the dataset using pandas.

3.Preprocess the data:

4.Drop unnecessary columns (e.g., CarName, car_ID).
5.Convert categorical variables using one-hot encoding.
6.Split the dataset into features (X) and target (Y), then into training and testing sets.

7.Standardize the features and target using StandardScaler.

8.Initialize the SGDRegressor model with appropriate parameters.

9.Train the model on the training data.

10.Predict the target values for the test data.

11.Evaluate the model using Mean Squared Error and R² score.

12.Display model coefficients and intercept.

13.Visualize actual vs predicted values with a scatter plot.

14.End of workflow. 
 

 

## Program:
~~~

/*
Program to implement SGD Regressor for linear regression.
Developed by: R K NIKKILVARSHAN
RegisterNumber: 212225040280
*/

# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv("CarPrice_Assignment.csv")
print(data.head())
print(data.info())

# Data preprocessing
# Dropping unnecessary columns and handling categorical variables
data = data.drop(['CarName', 'car_ID'], axis=1)
data = pd.get_dummies(data, drop_first=True)

# Splitting the data into features and target variable
X = data.drop('price', axis=1)
y = data['price']

scaler = StandardScaler()
# Standardizing the data
#scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(np.array(y).reshape(-1, 1)).flatten()

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the SGD Regressor model
sgd_model = SGDRegressor(max_iter=1000, tol=1e-3)

# Fitting the model on the training data
sgd_model.fit(X_train, y_train)

# Making predictions
y_pred = sgd_model.predict(X_test)

# Evaluating model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("="*50)
print("Mean Squared Error:", mse)
print("R² Score:", r2)
# Print evaluation metrics
print("Mean Squared Error:", mse)
print("R-squared Score:", r2)
print("="*50)

# Print model coefficients
print("Model Coefficients:")
print("Coefficients:", sgd_model.coef_)
print("Intercept:", sgd_model.intercept_)

# Visualizing actual vs predicted prices
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices using SGD Regressor")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Perfect prediction line
plt.show()
~~~
## Output:
![simple linear regression model for predicting the marks scored](sam.png)

<img width="1211" height="741" alt="Screenshot 2026-02-12 094612" src="https://github.com/user-attachments/assets/dc854d6e-7085-4ccb-9f0f-5e946f0b0c6c" />
<img width="1038" height="700" alt="Screenshot 2026-02-12 094625" src="https://github.com/user-attachments/assets/053a2425-94b1-428f-bd18-4cf5d3d5b64c" />
<img width="1107" height="510" alt="Screenshot 2026-02-12 094649" src="https://github.com/user-attachments/assets/f7073517-50a5-449b-9f43-4e52ddc24b75" />

<img width="1903" height="1010" alt="Screenshot 2026-02-12 091755" src="https://github.com/user-attachments/assets/9555c254-1835-4fe9-b3bb-b967f5120ef2" />


## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
