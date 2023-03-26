import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')
print(dataset)

matrix = dataset.iloc[:, :-1] # matrix from the dataset (Country, Age, Salary)
# iloc takes the no. of rows and columns from the data frame.
# dataset.iloc[:, :-1] the first element specifies the row and the last one in the dict represents the columns
dependant_variable = dataset.

