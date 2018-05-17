# Import the necessary modules and libraries
import math
import numpy as np
from sklearn import linear_model, datasets
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import pandas as pd

#Import data
weather_data = pd.read_csv('weather_madrid.csv')

weather_data.sort_values('Max TemperatureC', inplace=True)

weather_data.head()

x_columns = ['Max Humidity']

X = weather_data[x_columns]
y = weather_data['Max TemperatureC']

# Replace NaN with the mean in all Series
for column in X:
    mean = np.mean(X[X[column] == X[column]][column])
    X[X[column] != X[column]] = mean
    X[column] = pd.to_numeric(X[column])
    
mean_high_temp = np.mean(pd.to_numeric(y[y == y]))
y[y != y] = mean_high_temp
y_train = pd.to_numeric(y)

# Remove test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

#Verify there is no null values
print( 'Any null values? ', y_train.isnull().values.any() or X_train.isnull().values.any() )

#baseline value to beat; mean high temperature
#mean_high_temp
baseline_preds = []
for i in range(2044):
    baseline_preds.append(mean_high_temp)
baseline_preds = np.array( baseline_preds )

rmse_baseline = np.sqrt(mean_squared_error(y_test, baseline_preds))
print('Baseline RMSE: ', rmse_baseline)

tree_2 = DecisionTreeRegressor(max_depth=2)
rmse_scores = np.sqrt(-cross_val_score(tree_2, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
print("Tree RMSE (K-fold): %0.2f (+/- %0.2f)" % (rmse_scores.mean(), rmse_scores.std() * 2))

# Train the model using the training set
tree_2.fit(X_train, y_train)

#y predictions for all of our training examples
y_preds = tree_2.predict(X_test)

# Get RMSE for tree model
tree_2_rmse = np.sqrt(mean_squared_error(y_test, y_preds))

print('Tree RMSE (Test set): ', tree_2_rmse)

# Plot the results
plt.figure(figsize=(20,10))
# plt.figure()

#plot datapoints
# plt.scatter(X_test, y_preds, s=20, edgecolor="black",c="darkorange", label="data")
plt.plot(X_test, y_test, 'ro', c="red", label="data")

#plot our predictions
plt.plot(X_test, y_preds, 'ro', color="blue", label="Predictions")

#label axes
plt.xlabel("Max Humidity", size=24)
plt.ylabel("Max Temperature", size=24)
plt.title("Decision Tree Regression", size=24)

#show the graph
plt.legend()
plt.show()

tree_10 = DecisionTreeRegressor(max_depth=10)
rmse_scores = np.sqrt(-cross_val_score(tree_10, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
print("Tree RMSE (K-fold): %0.2f (+/- %0.2f)" % (rmse_scores.mean(), rmse_scores.std() * 2))

# Train the model using the training set
tree_10.fit(X_train, y_train)

#y predictions for all of our training examples
y_preds = tree_10.predict(X_test)

# Get RMSE for tree model
tree_rmse = np.sqrt(mean_squared_error(y_test, y_preds))

print('Tree RMSE (Test set): ', tree_rmse)

# Plot the results
plt.figure(figsize=(20,10))
# plt.figure()

#plot datapoints
# plt.scatter(X_test, y_preds, s=20, edgecolor="black",c="darkorange", label="data")
plt.plot(X_test, y_test, 'ro', c="red", label="data")

#plot our predictions
plt.plot(X_test, y_preds, 'ro', color="blue", label="Predictions")

#label axes
plt.xlabel("Max Humidity", size=24)
plt.ylabel("Max Temperature", size=24)
plt.title("Decision Tree Regression", size=24)

#show the graph
plt.legend()
plt.show()

# Create linear regression object
lmodel = linear_model.LinearRegression()

# Get Root Mean Squared Error via K Fold Cross Validation
rmse_scores = np.sqrt(-cross_val_score(lmodel, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
print("lmodel RMSE: %0.2f (+/- %0.2f)" % (rmse_scores.mean(), rmse_scores.std() * 2))

# Train the model using the training set
lmodel.fit(X_train, y_train)

#y predictions for all of our training examples
y_preds = lmodel.predict(X_test)

# Get RMSE for linear regression model
lmodel_rmse = np.sqrt(mean_squared_error(y_test, y_preds))
print('Linear Regression RMSE: ', lmodel_rmse)

# Plot the results
plt.figure(figsize=(20,10))
# plt.figure()

#plot datapoints
# plt.scatter(X_test, y_preds, s=20, edgecolor="black",c="darkorange", label="data")
plt.plot(X_test, y_test, 'ro', c="red", label="data")

#plot our predictions
plt.plot(X_test, y_preds, color="blue", label="Predictions", linewidth=2)

#label axes
plt.xlabel("Max Humidity", size=24)
plt.ylabel("Max Temperature", size=24)
plt.title("Decision Tree Regression", size=24)

#show the graph
plt.legend()
plt.show()

x_columns = [
    'Dew PointC', 'MeanDew PointC', 'Min DewpointC', 'Max Humidity', 'Mean Humidity', 'Min Humidity', \
    'Max Sea Level PressurehPa', 'Mean Sea Level PressurehPa', 'Min Sea Level PressurehPa', 'Max VisibilityKm', \
    'Mean VisibilityKm', 'Min VisibilitykM', 'Max Wind SpeedKm/h', 'Mean Wind SpeedKm/h', 'Precipitationmm', \
    'CloudCover', 'WindDirDegrees'
    ]

X = weather_data[x_columns]
y = weather_data['Max TemperatureC']

# Loop all Series
for column in X:
    mean = np.mean(X[X[column] == X[column]][column]) # mean of all non-NaN values
    X[X[column] != X[column]] = mean                  # replace all NaN with mean
    X[column] = pd.to_numeric(X[column])

# Replace NaN with mean in Target
mean_high_temp = np.mean(pd.to_numeric(y[y == y]))
y[y != y] = mean_high_temp
y_train = pd.to_numeric(y)

# Remove test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

#Verify there is no null values
print( 'Any null values? ', y_train.isnull().values.any() or X_train.isnull().values.any() )

#baseline value to beat; mean high temperature
#mean_high_temp
baseline_preds = []
for i in range(2044):
    baseline_preds.append(mean_high_temp)
baseline_preds = np.array( baseline_preds )

rmse_baseline = np.sqrt(mean_squared_error(y_test, baseline_preds))
print('Baseline RMSE: ', rmse_baseline)

tree_10 = DecisionTreeRegressor(max_depth=10)
rmse_scores = np.sqrt(-cross_val_score(tree_10, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
print("Tree RMSE (K-fold): %0.2f (+/- %0.2f)" % (rmse_scores.mean(), rmse_scores.std() * 2))

# Train the model using the training set
tree_10.fit(X_train, y_train)

#y predictions for all of our training examples
y_preds = tree_10.predict(X_test)

# Get RMSE for tree model
tree_10_rmse = np.sqrt(mean_squared_error(y_test, y_preds))

print('Tree RMSE (Test set): ', tree_10_rmse)

# Create linear regression object
lmodel = linear_model.LinearRegression()

# Get Root Mean Squared Error via K Fold Cross Validation
rmse_scores = np.sqrt(-cross_val_score(lmodel, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
print("lmodel RMSE: %0.2f (+/- %0.2f)" % (rmse_scores.mean(), rmse_scores.std() * 2))

# Train the model using the training set
lmodel.fit(X_train, y_train)

#y predictions for all of our training examples
y_preds = lmodel.predict(X_test)

# Get RMSE for linear regression model
lmodel_rmse = np.sqrt(mean_squared_error(y_test, y_preds))
print('Linear Regression RMSE: ', lmodel_rmse)

nn = MLPRegressor(solver='lbfgs', learning_rate_init=0.0001, max_iter=10000, #batch_size=100,
                     hidden_layer_sizes=(20), random_state=3, verbose=False)

rmse_scores = np.sqrt(-cross_val_score(nn, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
print("nn RMSE: %0.2f (+/- %0.2f)" % (rmse_scores.mean(), rmse_scores.std() * 2))

# Train the model using the training set
nn.fit(X_train, y_train)

#y predictions for all of our training examples
y_preds = nn.predict(X_test)

# Get RMSE for neural net model
nn_rmse = np.sqrt(mean_squared_error(y_test, y_preds))
print('NN RMSE: ', nn_rmse)