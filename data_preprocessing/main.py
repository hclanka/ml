import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('data.csv')

matrix = dataset.iloc[:, :-1].values # matrix from the dataset (Country, Age, Salary)
# iloc takes the no. of rows and columns from the data frame.
# dataset.iloc[:, :-1] the first element specifies the row and the last one in the dict represents the columns
dependant_variable = dataset.iloc[:, -1].values
# dependant variable is usually the last column of our dataset
# -1 is the last value of the number of columns (Purchased in the data)