#Multiple linear regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('C:\\Users\\Jonathan\\Documents\\MyProject\\MyPortfolio\\MachineLearning\\Part 2 - Regression\\Section 5 - Multiple Linear Regression\\50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#Encoding categorical data
#Encoding the Independent Variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])

onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

'''
Build a Model:

1. All in: use all your columns/data

2. Backward Elimination:

    i.      Select a significance level to stay in the model (e. g. SL = 0.05)
    ii.     Fit the full model with all possible predictors
    iii.    Consider the predictor with the highest P-Value. If P > SL, go to Step iv, otherwise to FIN
    iv.     Remove the predictor
    v.      Fit model without this variable
    vi.     Go back to step iii

    FIN: finish. Your model is ready

3. Forward Selection:
    i.      Select a significance level to enter the model (e. g. SL = 0.05)
    ii.     Fit all single regression models y ~ x_n. Select the one with the lowest P-value 
    iii.    Keep this variable and fit all possible models with one extra predictor added to the one(s) you already have
    iv.     Consider the predictor with the lowest P-value. If P < SL, go to Step iii, otherwise go to FIN

4. Bidirectional Elimination:

    i.      Select a significance level to enter and to stay in the model, e. g. SLENTER = 0.05 SLSTAY = 0.05
    ii.     Perform the next step of Forward Selection (new variables mut have: P < SLENTER to enter)
    iii.    Perform ALL steps of Bacward Elimination (old variables must have P < SLSTAY to stay)
    iv.     No new variables can enter and no old variables can exit
    v.      FIN

5. All possible models

    i.      Select a criterion of goodness of fit
    ii.     Construct all possible regression Models 2^N-1 total combinations
    iii.    Select the one with the best criterion
    iv.     FIN:Your Model is Ready
'''

print(X)