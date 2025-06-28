import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pickle

#loading the dataset
data=pd.read_csv("personality_dataset.csv")
data=pd.DataFrame(data)

#Encoding the data

Encoder=LabelEncoder()
data['Stage_fear']=Encoder.fit_transform(data['Stage_fear'])
data['Drained_after_socializing']=Encoder.fit_transform(data['Drained_after_socializing'])
data['Personality']=Encoder.fit_transform(data['Personality'])


#spliting the data into Two parts
x=data.iloc[:,:-1]
y=data['Personality']

#spliting the data into traning and testing

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#traning model                                                  #desicion_tre=91.03 -maxdep=2
model=LogisticRegression()
                                                                  #decision tree=99.67 max_depth=3
                                                                 # logistic=99.4
                                                                 #naive-guassin=99.6
                                                                 #naive-miltimial=
                                                                 #naive_bay-bernoli=
model.fit(x_train,y_train)                                       #Knn-n_neighbours=10 -acc=99.7
                                                                 #knn-n-neibours=5 acc=99.6
                                                                 #knn-n-neibouers=20 acc=99.8
                                                                 #svm-normal=99.4
                                                                 #svm -kernel-linear=99.4
  
  
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
                                                               #svm -kernel-poly=99.4
#        
# 
# 
# #sum-kernel-rbf=99.4
#                                                                  #sum-kernel-sigmoid=41.3
# #prediction
# y_pred=model.predict(x_test)
# print(y_pred)
# #accuracy
# accuracy=accuracy_score(y_test,y_pred)
# print(accuracy)
# #printing confucion matrix
# confusion=confusion_matrix(y_test,y_pred)
# # print(confusion)