import sklearn
from sklearn import datasets
from sklearn import svm
# import two packages for SVC (Support Vector Classifier)
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier 


cancer = datasets.load_breast_cancer()

#print ('Feature Names: ', cancer.feature_names)
#print ('Target Names: ' , cancer.target_names)

x = cancer.data
y = cancer.target

# split the data into train and test.
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split (x, y, test_size=0.2)
#print ('x_trains: ', x_train, 'y_train: ', y_train)

classes = ['malignant' 'benign']

# calssifer SVC (Supprt Vector Classifier) is part of svm (Support Vector Machine)
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# 'rbf': Radial Basis Function. Use linear instead unless you are the math expert.
# defualt accuracty iS 55%.
#clf = svm.SVC()
# lieanr model accuracy is 95%
#clf = svm.SVC(kernel='linear')
# polynomial model take very long to run. Do not try it unless it is very small data set.
#clf = svm.SVC(kernel='poly')
# polynomial model take very long to run. Set degree to 2. Runf for 20 seconds the accuracy = 96%
#clf = svm.SVC(kernel='poly', degree=2)
# lieanr model accuracy with C=2 (penalty term = 2) is 98%
#clf = svm.SVC(kernel='linear', C=2)
# lieanr model accuracy with C=1 (penalty term = 1) is 98%
#clf = svm.SVC(kernel='linear', C=1)

# you can try the KNeighborClassifier (K=9) model as well. Accuracy = 94%.
#clf = KNeighborsClassifier (n_neighbors=9)

# polynomial model take very long to run. Set degree to 2. Runf for 20 seconds the accuracy = 96%
clf = svm.SVC(kernel='poly', degree=2

# setup model
clf.fit (x_train, y_train)
y_pred = clf.predict (x_test)
# Use the metrics package
acc = metrics.accuracy_score (y_test, y_pred)
print ("acc: ", acc)