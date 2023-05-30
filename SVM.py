from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.svm import SVC
from utils.Dataset import read_dataset,flatten,plot25, PCAmethod
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [ 1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf', 'poly'],
              'degree': [9]
              }


#svm = SVC(kernel='rbf',C=10,gamma=0.1,degree=9,verbose=True)
svm = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)


dim=200
n=30
train_data1,train_labels=read_dataset('train',dim)
train_data=flatten(train_data1)
test_data,test_labels=read_dataset('test',dim)
test_data=flatten(test_data)
pca = PCA(n_components=n)
train_data, test_data = pca.fit_transform(train_data), pca.transform(test_data)

#train_data, test_data=PCAmethod(train_data,test_data,66)

svm.fit(train_data, train_labels)
pred = svm.predict(test_data)
print('SVM test accuracy', accuracy_score(test_labels, pred)) #Accuracy
confusion=confusion_matrix(test_labels,pred)