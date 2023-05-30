from sklearn.neighbors import KNeighborsClassifier
#from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score,confusion_matrix
from utils.Dataset import read_dataset, flatten
#from sklearn import datasets
#from skimage import exposure
import numpy as np
#import imutils
#import cv2
#import sklearn
from sklearn.decomposition import PCA


dim = 200
train_data1,train_labels=read_dataset('train',dim)
train_data=flatten(train_data1)
test_data,test_labels=read_dataset('test',dim)
test_data=flatten(test_data)
n=30
pca = PCA(n_components=n)
train_data, test_data = pca.fit_transform(train_data), pca.transform(test_data)
k=33
model = KNeighborsClassifier(n_neighbors=k, weights='uniform', algorithm='auto', leaf_size=200, p=2, metric='minkowski', metric_params=None, n_jobs=None)
model.fit(train_data, train_labels)

pred = model.predict(test_data)

print( "KNN test accuracy" , accuracy_score(test_labels, pred))
confusion= confusion_matrix(test_labels,pred)