import sklearn
from sklearn import datasets
from sklearn import svm

cancer = datasets.load_breast_cancer()

print ('Feature Names: ', cancer.feature_names)
print ('Target Names: ' , cancer.target_names)

x = cancer.data
y = cancer.target

# split the data into train and test.
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split (x, y, test_size=0.2)
print ('x_trains: ', x_train, 'y_train: ', y_train)

classes = ['malignant' 'benign']