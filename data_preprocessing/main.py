import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer

#################################
##### Importing the Dataset #####
#################################

dataset = pd.read_csv('data.csv')

matrix = dataset.iloc[:, :-1].values # matrix from the dataset (Country, Age, Salary)
# iloc takes the no. of rows and columns from the data frame.
# dataset.iloc[:, :-1] the first element specifies the row and the last one in the dict represents the columns
dependant_variable = dataset.iloc[:, -1].values
# dependant variable is usually the last column of our dataset
# -1 is the last value of the number of columns (Purchased in the data)

"""
the dataset above is split into 2 different entities, these 2 entities are the matrix and dependant variable.
these 2 entities are used for the machine learning model as inputs
"""

###########################################
##### Taking Care Of The Missing Data #####
###########################################

# the simple imputer object is used for replacing the missing values with the specific method of replacement
# methods include the mode of the data, median of the data and mean which is the most generic
# replacing the missing values with relevant data can increase the accuracy of our model
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
# the fit function here takes the specific areas of the dataset which need to be replaced with a calculated value
imputer.fit(matrix[:, 1:3])
# the transform function returns the calculated values into the matrix, and we set the values we get into the missing value slots
matrix[:, 1:3] = imputer.transform(matrix[:, 1:3])
