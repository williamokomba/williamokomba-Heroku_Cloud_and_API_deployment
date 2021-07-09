
# creating our model
import sklearn
#imporing the necessary libruaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

#
#loading and reading the dataset
hiring_df = pd.read_csv('hiring.csv')


#Building the model
#importing the necessary file
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#splitting dataset into target and feture variables
x = hiring_df[['experience', 'test_score', 'interview_score']] # feature variable
y= hiring_df['salary']# target variable

#splitting data into train and test datasets ( we will use 70% of data as train data)
x_train, y_train, y_train, y_test= train_test_split(x,y, test_size = 0.3, random_state=101)

#building and training the model
regressor= LinearRegression()
regressor.fit(x_train, y_train)

#saving the model in the disk
pickle.dump(regressor, open('model.pkl','wb'))







