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

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Reegression the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train) 

#Predicting the best set results
y_pred = regressor.predict(X_test)

#Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm

#Adding the b_0 factor using a (1) column  (f = b_0 + b_1*x_1...)
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)

#Complete Model
X_opt = X[:, [0, 1, 2, 3, 4, 5]]

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# First iteration
X_opt = X[:, [0, 1, 3, 4, 5]]

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# Second Iteration
X_opt = X[:, [0, 3, 4, 5]]

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

#Third Iteration
X_opt = X[:, [0, 3, 5]]

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

#Fourth Iteration
X_opt = X[:, [0, 3]]

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

print(regressor_OLS.summary())

'''
Automatic model:

import statsmodel.formula.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)

    regressor_OLS.summary()
    return x


SL = 0.05
X_opt = X[:, ...]
X_Modeled = bacwardElimination(X_opt, SL)

---------------------------------------------------------------------------------------------

Model with P and R-Squared:

import statsmodels.formula.api as sm
def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x

'''

print(X)