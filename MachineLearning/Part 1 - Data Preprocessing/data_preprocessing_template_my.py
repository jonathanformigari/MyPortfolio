import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset 

csvFileName = "C:\\Users\\Jonathan\\Documents\\MyProject\\MyPortfolio\\MachineLearning\\Part 1 - Data Preprocessing\\Data.csv"

dataset = pd.read_csv(csvFileName)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#Importing lib to preprocessing and missing data
# from sklearn.preprocessing import Imputer
# imputer = Imputer(missing_values="NaN", strategy="mean", axis=0) #Creation of the object
# imputer = imputer.fit(X[:, 1:3]) #Using object's method "fit" to fit where we have to insert missing data
# X[:, 1:3] = imputer.transform(X[:, 1:3]) #Using object's method "transform" to get transformed data 

#Encoding categorical data
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# labelencoder_X = LabelEncoder()
# X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

# onehotencoder = OneHotEncoder(categorical_features=[0])
# X = onehotencoder.fit_transform(X).toarray()

# labelencoder_y = LabelEncoder()
# y = labelencoder_y.fit_transform(y)

#Spliting my set into the Training set and the Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train) #need to fit and transform
# X_test = sc_X.transform(X_test) #did not need fit again

#Do we have to scale dummy variables? It depends on how much we want to keep our model
#We do not have to scale y in this case