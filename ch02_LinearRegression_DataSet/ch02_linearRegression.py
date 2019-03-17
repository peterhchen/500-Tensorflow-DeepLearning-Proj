import pandas as pd;
import numpy as np;
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv("student/student-mat.csv", sep=";")
print(data.head())
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
print(data.head())

predict = "G3"
# we drop "G3" label data. These are all training data.
x = np.array(data.drop([predict], 1))
# We get all of the features and attribute data
y = np.array(data[predict])
# split into 4 different array: x_train is sectoin of x, y_train is section of y.
# x_test and y_test are used to test the model accuracy.
# We spli tout 10% of data for test sample.
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split (x, y, test_size=0.1)  