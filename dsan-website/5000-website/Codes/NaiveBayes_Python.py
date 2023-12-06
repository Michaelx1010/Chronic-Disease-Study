import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the Framingham Heart Study data set
data = pd.read_csv("data/frmgham2.csv")
data.head()
data.describe

# Missing data handling
print(data.isna().sum())
#data.dropna()

# Removing data variables that have 30% or more of missing data
# Calculate the percentage of missing data in each column
missing_percentage = (data.isna().sum() / len(data)) * 100

# Set the threshold for missing data (30%)
threshold = 30

# Identify columns with missing data exceeding the threshold
columns_to_remove = missing_percentage[missing_percentage > threshold].index
print(columns_to_remove)

# Remove the identified columns from the DataFrame
data = data.drop(columns=columns_to_remove)
data.dropna(inplace=True)

data.head()

# Split data into train/test/validation sets
# Split target variables and predictor variables
X = data.drop('CVD', axis=1)
y = data['CVD']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42) 

X_train.head()

# The objective for splitting data into train/test/validation sets is to prevent model from overfitting and improve generalization of the model.

# Feature selection using ANOVA F-test
s= SelectKBest(score_func=f_classif, k=25)  
X_new = s.fit_transform(X, y)
sf = X.columns[s.get_support()]
print("Selected features:", sf)

# Model training

# Training sets results
# Train the Naive Bayes model using selected features
nb_model = GaussianNB()
nb_model.fit(X_train[sf], y_train)

# Obtain evaluation metrics for both training sets/validation sets
y_train_pred = nb_model.predict(X_train[sf])
y_val_pred = nb_model.predict(X_val[sf])

val_accuracy = accuracy_score(y_val, y_val_pred)
train_accuracy = accuracy_score( y_train, y_train_pred)

print(f"Accuracy for training set: {train_accuracy}")
print(classification_report( y_train, y_train_pred))

# Validation sets results
print(f"Accuracy for validation set: {val_accuracy}")
print(classification_report(y_val, y_val_pred))

# Confusion Matrix for Validation Data
cm_val = confusion_matrix(y_val, y_val_pred)
print(cm_val)

# Interpretation: 
# The Naive Bayes classifier works really well when tested, showing high levels of correct predictions and a good balance in its ability to identify true cases of the condition it's predicting. The F1-scores, which combine precision and recall, are also high, indicating the model is consistent in its predictions. Moreover, the model is slightly more accurate on the validation data than on the data it was trained on, which suggests its generalization when used in real-world data. Also, the high accuracy score on validation sets also indicates that the model neither over-fitting nor under-fitting, but reaching a balance between the two. Overall, these results highlight the model as a dependable tool for predictive modelling.

# Visualizations

# Confusion matrix heatmap
# Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
plt.title('Confusion Matrix for Validation Data')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# The confusion matrix heatmap is a visual interpretation of the fitted model.

# Accuracy evaluation bar plots
# Metrics for the training set
ta = accuracy_score(y_train, y_train_pred)
tp = precision_score(y_train, y_train_pred)
tr = recall_score(y_train, y_train_pred)
tf1 = f1_score(y_train, y_train_pred)

# Metrics for the validation set
va = accuracy_score(y_val, y_val_pred)
vp = precision_score(y_val, y_val_pred)
vr = recall_score(y_val, y_val_pred)
vf1 = f1_score(y_val, y_val_pred)

# Bar plot of accuracy metrics
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Training': [ta, tp, tr, tf1],
    'Validation': [va, vp, vr, vf1]
})

metrics_df.set_index('Metric', inplace=True)
metrics_df.plot.bar(rot=0, figsize=(10, 6))
plt.title('Comparison of Training and Validation Metrics')
plt.ylabel('Score')
plt.show()

# The bar plot demonstrates comparison of scores of Accuracy, precision, recall, and F-1 score between training and validation models, with higher scores on every metric of the validation models, suggesting that the model learns well on the training data and also performs strongly on unseen data, suggesting a good model performance.

# ROC curve
# Predict probabilities for the validation set
yp = nb_model.predict_proba(X_val[sf])[:, 1]

# Compute ROC curve and ROC area for the validation set
fpr, tpr, _ = roc_curve(y_val, yp)
roc = auc(fpr, tpr)

# Plot ROC Curve for the validation set
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Validation Set')
plt.legend(loc="lower right")
plt.show()