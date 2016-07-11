import pandas as pd

# import the data from the dataset
#patients=pd.read_csv('data_p.csv')
hospitals=pd.read_csv('data_h.csv')
#names_p=set()
names_h=set()

# for index,row in patients.iterrows():
#     names_p.add(row['FACILITY_NAME'])
for index,row in hospitals.iterrows():
    names_h.add(row['FAC_NAME'])
#print (names_p-names_h)

# import random forest classfier as present in the aklearn package
from sklearn.ensemble import RandomForestClassifier
# Assign the F9 column as the target
target=hospitals['F9']
# Delete the features NOT included withint the random forest
hospitals.drop('F9',axis=1,inplace=True)
hospitals.drop('OSHPD_ID',axis=1,inplace=True)
hospitals.drop('FAC_NAME',axis=1,inplace=True)
# initialize a random forest classifier with 10 estimators--10 decisiont trees would be constructed
rf=RandomForestClassifier(n_estimators=10)

# import the function to split the training & testing set
from sklearn.cross_validation import train_test_split
# creating the training & testing dataset. test_size=0.2 -- 20% would be used as the testing data
# split based on train_test_split(features,labels,test_size=XX)
# X_train & X_test--> subsets of features;
# y_train & y_test --> subsets of labels;
# corresponding to the X_train/test[0] --> y_train/test[0] holds the associated label for the X sets
X_train, X_test, y_train, y_test=train_test_split(hospitals,target,test_size=0.1)
# initiate the fit function and start the random forest
rf.fit(X_train,y_train)
# used the created forest to predict the labels of the test set
predicted=rf.predict(X_test)

# in order to match the predictions with the actual values & check how much accuracy we got, have to use the accuracy score function
from sklearn.metrics import accuracy_score
print "\nSingle Random Forest Score: ", accuracy_score(y_test,predicted)

from sklearn.cross_validation import cross_val_score
scores=cross_val_score(rf,hospitals,target,cv=10)
print "\nRandom Forest SCORES FOR 10 FOLDS cross validation \n", scores
print "\nMEAN SCORE for Random Forest", scores.mean()

# Import the SVM classification associated package
from sklearn.svm import SVC
clf=SVC()
# Specify the kind of SVC. Here I employed rbf so as to make the calculation more flexible
estimator1=SVC(kernel="rbf")
scores_svm=cross_val_score(estimator1,hospitals,target,cv=10)
print "\nSVM SCORES FOR 10 FOLDS cross validation\n", scores_svm
print "\nMEAN SCORE for SVM", scores_svm.mean()

# visualization
from sklearn.metrics import confusion_matrix
# each row-each class of the label
print "\nConfusion matrix regarding y_test vs predicted\n",confusion_matrix(y_test,predicted)

# Reducing the dimensions and plotting them
from sklearn.manifold import TSNE
model=TSNE(n_components=2)
embed=model.fit_transform(hospitals,target)
print embed

hospitals['F10'].plot(kind="hist")
