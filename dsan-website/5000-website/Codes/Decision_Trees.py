# Importing necessary libraries
import sklearn
from sklearn import datasets
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

# Load the Framingham Heart Study dataset for classification
data = pd.read_csv("data/frmgham2.csv")
data.head()

# Split target variables and predictor variables
y = data['CVD']
x = data.drop('CVD', axis=1)
x = x.iloc[:, 1:]
x = x.values
y = y.values

# Print class distribution for 'CVD'
cd = data['CVD'].value_counts(normalize=True) * 100
print(cd)

# Exploratory Data Analysis for classification
data_eda = data.iloc[:, 1:]
print(data_eda.describe())

# Calculate correlation and plot heatmap for classification
corr = data.corr()
sns.set_theme(style="white")
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, cmap=cmap, vmin=-1, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show();

# Random classifier function for classification
def random_classifier(y_data):
    ypred = []
    max_label = np.max(y_data)
    for i in range(0, len(y_data)):
        ypred.append(int(np.floor((max_label + 1) * np.random.uniform(0, 1))))

    print("-----RANDOM CLASSIFIER-----")
    print("count of prediction:", Counter(ypred).values())
    print("probability of prediction:", np.fromiter(Counter(ypred).values(), dtype=float) / len(y_data))
    print("accuracy", accuracy_score(y_data, ypred))
    print("percision, recall, fscore,", precision_recall_fscore_support(y_data, ypred))

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Call random classifier function for classification
random_classifier(y)

# Splitting the data into training and testing sets for classification
test_ratio = 0.2
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_ratio, random_state=0)
y_train = y_train.flatten()
y_test = y_test.flatten()

# Print shapes of training and testing sets for classification
print("x_train.shape        :", x_train.shape)
print("y_train.shape        :", y_train.shape)
print("X_test.shape     :", x_test.shape)
print("y_test.shape     :", y_test.shape)

# Lists to store hyperparameters and errors for classification
hyper_param = []
train_error = []
test_error = []

# Hyperparameter tuning for max_depth for classification
for i in range(1, 40):
    model = DecisionTreeClassifier(max_depth=i)
    model.fit(x_train, y_train)
    yp_train = model.predict(x_train)
    yp_test = model.predict(x_test)
    err1 = mean_absolute_error(y_train, yp_train)
    err2 = mean_absolute_error(y_test, yp_test)
    hyper_param.append(i)
    train_error.append(err1)
    test_error.append(err2)
    if (i == 1 or i % 10 == 0):
        print("hyperparam =", i)
        print(" train error:", err1)
        print(" test error:", err2)

# Plot training and testing errors for max_depth for classification
plt.plot(hyper_param, train_error, linewidth=2, color='k')
plt.plot(hyper_param, test_error, linewidth=2, color='b')

plt.xlabel("Depth of tree (max depth)")
plt.ylabel("Training (black) and test (blue) MAE (error)")

i = 1
print(hyper_param[i], train_error[i], test_error[i])

# Lists to store hyperparameters and errors for classification
hyper_param = []
train_error = []
test_error = []

# Hyperparameter tuning for min_samples_split for classification
for i in range(2, 100):
    model = DecisionTreeClassifier(min_samples_split=i)
    model.fit(x_train, y_train)
    yp_train = model.predict(x_train)
    yp_test = model.predict(x_test)
    err1 = mean_absolute_error(y_train, yp_train)
    err2 = mean_absolute_error(y_test, yp_test)
    hyper_param.append(i)
    train_error.append(err1)
    test_error.append(err2)
    if (i % 10 == 0):
        print("hyperparam =", i)
        print(" train error:", err1)
        print(" test error:", err2)

# Plot training and testing errors for min_samples_split for classification
plt.plot(hyper_param, train_error, linewidth=2, color='k')
plt.plot(hyper_param, test_error, linewidth=2, color='b')

plt.xlabel("Minimum number of points in split (min_samples_split)")
plt.ylabel("Training (black) and test (blue) MAE (error)")

# Create a Decision Tree Classifier model for classification
model = DecisionTreeClassifier(max_depth=10, min_samples_split=30)
model.fit(x_train, y_train)

# Predictions on training and testing data for classification
yp_train = model.predict(x_train)
yp_test = model.predict(x_test)

# Calculate Mean Absolute Error for training and testing data for classification
err1 = mean_absolute_error(y_train, yp_train)
err2 = mean_absolute_error(y_test, yp_test)

# Print training and testing errors for classification
print(" train error:", err1)
print(" test error:", err2)

# Plot actual vs. predicted values for both training and testing data for classification
plt.plot(y_train, yp_train, "o", color='k')
plt.plot(y_test, yp_test, "o", color='b')
plt.plot(y_train, y_train, "-", color='r')

plt.xlabel("y_data")
plt.ylabel("y_pred (blue=test)(black=Train)")

# Create a Decision Tree plot for classification
def plot_tree(model):
    fig = plt.figure(figsize=(25, 20))
    _ = tree.plot_tree(model, filled=True)
    plt.show()

# Plot the Decision Tree for classification
plot_tree(model)

# Load the Framingham Heart Study dataset for regression
data = pd.read_csv("data/frmgham2.csv")
data.head()

# Remove rows with missing values
data.dropna(inplace=True)

# Split target variable 'GLUCOSE' and predictor variables for regression
y = data['GLUCOSE']
x = data.drop('GLUCOSE', axis=1)
x = x.iloc[:, 1:]
x = x.values
y = y.values

# Normalize the data for regression
x = 0.1 + (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0))
y = 0.1 + (y - np.min(y, axis=0)) / (np.max(y, axis=0) - np.min(y, axis=0))

# Exploratory Data Analysis for regression
data_eda = data.iloc[:, 1:]
print(data_eda.describe())

# Calculate correlation and plot heatmap for regression
corr = data.corr()
sns.set_theme(style="white")
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, cmap=cmap, vmin=-1, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show();

# Splitting the data into training and testing sets for regression
test_ratio = 0.2
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_ratio, random_state=0)
y_train = y_train.flatten()
y_test = y_test.flatten()

# Print shapes of training and testing sets for regression
print("x_train.shape        :", x_train.shape)
print("y_train.shape        :", y_train.shape)
print("X_test.shape     :", x_test.shape)
print("y_test.shape     :", y_test.shape)

# Lists to store hyperparameters and errors for regression
hyper_param = []
train_error = []
test_error = []

# Hyperparameter tuning for max_depth for regression
for i in range(1, 40):
    model = DecisionTreeRegressor(max_depth=i)
    model.fit(x_train, y_train)
    yp_train = model.predict(x_train)
    yp_test = model.predict(x_test)
    err1 = mean_absolute_error(y_train, yp_train)
    err2 = mean_absolute_error(y_test, yp_test)
    hyper_param.append(i)
    train_error.append(err1)
    test_error.append(err2)
    if (i == 1 or i % 10 == 0):
        print("hyperparam =", i)
        print(" train error:", err1)
        print(" test error:", err2)

# Plot training and testing errors for max_depth for regression
plt.plot(hyper_param, train_error, linewidth=2, color='k')
plt.plot(hyper_param, test_error, linewidth=2, color='b')

plt.xlabel("Depth of tree (max depth)")
plt.ylabel("Training (black) and test (blue) MAE (error)")

i = 1
print(hyper_param[i], train_error[i], test_error[i])

# Lists to store hyperparameters and errors for regression
hyper_param = []
train_error = []
test_error = []

# Hyperparameter tuning for min_samples_split for regression
for i in range(2, 100):
    model = DecisionTreeRegressor(min_samples_split=i)
    model.fit(x_train, y_train)
    yp_train = model.predict(x_train)
    yp_test = model.predict(x_test)
    err1 = mean_absolute_error(y_train, yp_train)
    err2 = mean_absolute_error(y_test, yp_test)
    hyper_param.append(i)
    train_error.append(err1)
    test_error.append(err2)
    if (i % 10 == 0):
        print("hyperparam =", i)
        print(" train error:", err1)
        print(" test error:", err2)

# Plot training and testing errors for min_samples_split for regression
plt.plot(hyper_param, train_error, linewidth=2, color='k')
plt.plot(hyper_param, test_error, linewidth=2, color='b')

plt.xlabel("Minimum number of points in split (min_samples_split)")
plt.ylabel("Training (black) and test (blue) MAE (error)")

# Create a Decision Tree Regressor model for regression
model = DecisionTreeRegressor(max_depth=1, min_samples_split=50)
model.fit(x_train, y_train)

# Predictions on training and testing data for regression
yp_train = model.predict(x_train)
yp_test = model.predict(x_test)

# Calculate Mean Absolute Error for training and testing data for regression
err1 = mean_absolute_error(y_train, yp_train)
err2 = mean_absolute_error(y_test, yp_test)

# Print training and testing errors for regression
print(" train error:", err1)
print(" test error:", err2)

# Plot actual vs. predicted values for both training and testing data for regression
plt.plot(y_train, yp_train, "o", color='k')
plt.plot(y_test, yp_test, "o", color='b')
plt.plot(y_train, y_train, "-", color='r')

plt.xlabel("y_data")
plt.ylabel("y_pred (blue=test)(black=Train)")

# Create a Decision Tree plot for regression
def plot_tree(model):
    fig = plt.figure(figsize=(25, 20))
    _ = tree.plot_tree(model, filled=True)
    plt.show()

# Plot the Decision Tree for regression
plot_tree(model)