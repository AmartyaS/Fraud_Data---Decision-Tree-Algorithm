# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 04:43:58 2021

@author: ASUS
"""
#Importing all the library files
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

#Loading up the data file
file=pd.read_csv("D:\Data Science Assignments\R-Assignment\Decision_Tree\Fraud_check.csv")

#Data Manipulation also used Hot En-Coding Method
data=file
data['Taxable.Income']=["Risky" if x<=30000 else "Good" for x in file['Taxable.Income']]
data['Undergrad']=[1 if x=='YES' else 0 for x in file['Undergrad']]
data['Urban']=[1 if x=='YES' else 0 for x in file['Urban']]
data['Undergrad']=[1 if x=='YES' else 0 for x in file['Undergrad']]
data=pd.get_dummies(data,columns=["Marital.Status"],prefix=["Status"])
data['Taxable.Income'].value_counts()
data.dtypes

#Defining the target and the predictor columns
colnames=list(data)
colnames
target=colnames[1]
predictor=colnames[0:1]+colnames[2:8]

#Splitting the data into training and testing dataset
train,test=train_test_split(data,test_size=0.2)
train['Taxable.Income'].value_counts()
test['Taxable.Income'].value_counts()

#Formation of Model
model=DecisionTreeClassifier(criterion='entropy')
model.fit(train[predictor],train[target])

#Decision Tree plot
plot_tree(model)

#Checking the accuracy of the model
preds=model.predict(test[predictor])
pd.Series(preds).value_counts()
pd.crosstab(test[target],preds)
np.mean(preds==test['Taxable.Income'])
