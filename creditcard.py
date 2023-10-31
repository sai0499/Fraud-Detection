import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyodbc as dbc
import sys
import scipy

# load the dataset using pandas
conn = dbc.connect('Driver={SQL Server};''Server=HARISH;''Database=harish;''Trusted_Connection=yes;')
sqlexec = conn.cursor()

# execute the SQL query
sqlexec.execute('SELECT * FROM dbo.creditcard')

# fetch the data from the cursor and convert to pandas dataframe
data = pd.DataFrame.from_records(sqlexec.fetchall(), columns=[desc[0] for desc in sqlexec.description])

# dataset exploring
print(data.columns)

# Print the shape of the data
data = data.sample(frac=0.1, random_state=1)
print(data.shape)
 # Print the first few rows of the data
print(data.head())

data.columns=['Time','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16',
              'V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28','Amount','Class']

# remove the double-quote characters from the "Class" column
data['Class'] = data['Class'].str.replace('"', '')

# convert the "Class" column to integers
data['Class'] = data['Class'].astype(int)

# V1 - V28 are the results of a PCA Dimensionality reduction to protect user identities and sensitive features

# Plot histograms of each parameter 
data.hist(figsize=(20, 20))
plt.show()

# Determine number of fraud cases in dataset

Fraud = data[data['Class'] == 1]
Valid = data[data['Class'] == 0]


outlier_fraction = len(Fraud) / float(len(Valid))
print(outlier_fraction)


print('Fraud Cases: {}'.format(len(data[data['Class'] == 1])))
print('Valid Transactions: {}'.format(len(data[data['Class'] == 0])))

# Correlation matrix
corrmat = data.corr()
fig = plt.figure(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()

# Get all the columns from the dataFrame
columns = data.columns.tolist()

 # Filter the columns to remove data we do not want
columns = [c for c in columns if c not in ["Class"]]

 # Store the variable we'll be predicting on
target = "Class"

X = data[columns]
Y = data[target]

 # Print shapes
print(X.shape)
print(Y.shape)

from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# define random states
state = 13

# define outlier detection tools to be compared
classifiers = {
    "Isolation Forest": IsolationForest(max_samples=len(X),
                                        contamination=outlier_fraction,
                                        random_state=state),
    "Local Outlier Factor": LocalOutlierFactor(
        n_neighbors=20,
        contamination=outlier_fraction)}
plt.figure(figsize=(9, 7))

n_outliers = len(Fraud)
n_outliers

for i, (clf_name, clf) in enumerate(classifiers.items()):

    # fit the data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)

        # Reshape the prediction values to 0 for valid, 1 for fraud. 
        y_pred[y_pred == 1] = 0
        y_pred[y_pred == -1] = 1
        n_errors = (y_pred != Y).sum()

        # Run classification metrics
        print('{}: {}'.format(clf_name, n_errors))
        print(accuracy_score(Y, y_pred))
        print(classification_report(Y, y_pred))
