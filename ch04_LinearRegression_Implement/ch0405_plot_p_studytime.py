import pandas as pd;
import numpy as np;
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot;
import pickle;
from matplotlib import style;


data = pd.read_csv("student/student-mat.csv", sep=";")
#print(data.head())
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
print(data.head())

predict = "G3"
# we drop "G3" label data. These are all training data.
x = np.array(data.drop([predict], 1))
# We get all of the features and attribute data
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split (x, y, test_size=0.1)

"""
best = 0
for _ in range (30):
    # split into 4 different array: x_train is sectoin of x, y_train is section of y.
    # x_test and y_test are used to test the model accuracy.
    # We spli tout 10% of data for test sample.
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split (x, y, test_size=0.1)


    linear = linear_model.LinearRegression()
    # create a model
    linear.fit(x_train, y_train)
    # How well of the model doing. Or how well of algorithm working. 
    acc = linear.score (x_test, y_test)
    # Print how accurate of the score.
    print (acc)

    if (acc > best):
        # Save the model
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)
"""

pickle_in = open ("studentmodel.pickle", "rb")
linear = pickle.load (pickle_in)

# Print coefficeint.
print ('Coefficient: ', linear.coef_)
# Print intercept.
print ('Intercept: ', linear.intercept_)
# print all of the prediction
predictions = linear.predict (x_test)
for x in range (len (predictions)):
    print (predictions[x], x_test[x], y_test[x])

# p = "G1"
# p = "G2"
p = "studytime"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel (p)
pyplot.ylabel ("Final Grade")
pyplot.show()
