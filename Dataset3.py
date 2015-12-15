__author__ = 'harshad'

''''
About script:
    Experimenting with different inbuilt classfication techniques of Sklearn and using ensemble methods
    to work on a dataset consisting very few samples and a large number of features
'''

import numpy as np
import scipy
from scipy import sparse
import math
import sklearn
import numpy as np
from sklearn.cross_validation import KFold
from sklearn import preprocessing
import sys
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt
from DecisionTree import *
from Node import *
from copy import deepcopy

from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingClassifier


matrix = np.loadtxt('dataset3/train.txt')
truths = np.loadtxt('dataset3/train_truth.txt')

matrix = matrix.transpose()
print matrix.shape
pca = PCA(n_components=10)
pca.fit(matrix)
matrix = pca.transform(matrix)
print 'new mat shape',matrix.shape
print 'truths shape =',truths.shape

test_sample = np.loadtxt('dataset3/test.txt')
test_sample = test_sample.transpose()
print 'test sample = ',test_sample.shape
origmat = np.loadtxt('dataset3/train.txt')
origmat = origmat.transpose()
origmat = np.vstack((origmat,test_sample))
pca = PCA(n_components=10)
pca.fit(origmat)
origmat = pca.transform(origmat)

matrix = origmat[:35,:]
test_matrix = origmat[35:,:]
print 'new mat shape',matrix.shape
print 'test shape',test_matrix.shape


index = len(matrix) * 0.6
X_train, X_test, y_train, y_test = train_test_split(matrix, truths, test_size=0.5, random_state=0)

'''use SVM'''
clf = svm.SVC()
clf.fit(X_train,y_train)
print 'score for svm = ', clf.score(X_test,y_test)
print 'predicted = ', clf.predict(test_matrix)

'''use LogReg'''
clf = LogisticRegression()
clf.fit(X_train,y_train)
print 'score for logistic = ', clf.score(X_test,y_test)
print'logistic rpediction= ',clf.predict(test_matrix)

'''Use DecTree'''
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(X_train,y_train)
print 'score for dtree= ', clf.score(X_test,y_test)
print'dtrr rpediction= ',clf.predict(test_matrix)

'''Use RanForest'''
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=15)
clf.fit(X_train,y_train)
print 'score for rfor= ', clf.score(X_test,y_test)
print'rfor rpediction= ',clf.predict(test_matrix)

'''use Adaboost '''
clf = AdaBoostClassifier(n_estimators=10)
clf.fit(X_train,y_train)
print 'score for adaboost = ', clf.score(X_test,y_test)
print'adaboost rpediction= ',clf.predict(test_matrix)

'''use Gbc'''
clf = GradientBoostingClassifier(n_estimators=10)
clf.fit(X_train,y_train)
print 'score for gdbt = ', clf.score(X_test,y_test)
print'gdbt rpediction= ',clf.predict(test_matrix)
