# This specifies the
import pandas as pd
from sklearn.svm import SVC,LinearSVC
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression as lr
from sklearn.cross_validation import cross_val_score as cv1
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

#read the dataset into a dataframe
data=pd.read_csv('data_h.csv')
#convert the categorical variables into dummy variables.
data=pd.get_dummies(data)

#Select which features are most important.
estimator = SVC(kernel="linear")
selector = RFECV(estimator, step=1, cv=10)
y=data['F19']
data.drop('F19', axis=1, inplace=True)
selector = selector.fit(data, y)

#print which features have been selected
print "ATTRIBUTES WHICH HAVE BEEN SELECTED\n"
for i in xrange(0,len(data.columns)):
	if(selector.support_[i]==True):
		print data.columns[i]

df1=data[['FAC_NAME','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13','F14','F15','F16','F17','F18','F19','F20','F21','F22']]
clf=SVC()  #???
scores=cv1(clf,df1,y,cv=10)
print "\nSVC Cross validated Scores:\n"
print scores

clf1=lr()
scores1=cv1(clf1,df1,y,cv=10)
print "\nLogistic Regression Cross validated Scores:\n"
print scores1

model = GaussianNB()
scores2=cv1(model,df1,y,cv=10)
print "\nNaive Bayes Cross validated Scores:\n"
print scores2

model = DecisionTreeClassifier()
scores3=cv1(model,df1,y,cv=10)
print "\nDecision Trees validated Scores:\n"
print scores3

clf=LinearSVC()
scores4=cv1(clf,df1,y,cv=10)
print "\nLinear SVC Cross validated Scores:\n"
print scores4

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
model = GridSearchCV(lr(), param_grid, cv=10)
model.fit(df1,y)
print "\nBEST SCORE WITH GRID SEARCH LogisticRegression:"
print model.best_score_

param_grid = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
model1 = GridSearchCV(SVC(), param_grid, cv=10)
model1.fit(df1,y)
print "\nBEST SCORE WITH GRID SEARCH SVC:"
print model1.best_score_
