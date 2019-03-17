import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv ("car.data")
print(data.head())

le = preprocessing.LabelEncoder()
buying = le.fit_transform (list (data["buying"]))
maint = le.fit_transform (list (data["maint"]))
door = le.fit_transform (list (data["door"]))
persons = le.fit_transform (list (data["persons"]))
lug_boot = le.fit_transform (list (data["lug_boot"]))
safety = le.fit_transform (list (data["safety"]))
cls = le.fit_transform (list (data["class"]))
print ('buying: ', buying); print ('maint: ', maint); print ('door: ', door); print ('persons: ', persons); 
print ('lug_boot: ', lug_boot); print ('safety: ', safety); print ('cls: ', cls)
predict = "class"
X = list (zip (buying, maint, door, persons, lug_boot, safety))
y = list (cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split (X, y, test_size = 0.1)
print ('x_train:', x_train); print ('y_train: ', y_train); print ('x_test: ', x_test); print ('y_test: ', y_test)