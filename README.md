# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the required libraries
2.Upload and read the dataset.
3.Check for any null values using the isnull() function.
4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.
```
## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Kavinesh M
RegisterNumber: 212222230064 
*/
import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
X=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
X.head()
Y=data["left"]
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(X_train,Y_train)
Y_pred=dt.predict(X_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(Y_test,Y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
### data.head()
![head](https://github.com/kavinesh8476/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118466561/7053acea-9767-4f0d-be48-4020977d5099)

### data.info()
![info](https://github.com/kavinesh8476/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118466561/2f04c5df-7532-4eed-b66d-da24ada95ce8)

### isnull() and sum()
![null](https://github.com/kavinesh8476/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118466561/6ee3fbbd-f898-4f3b-8c9a-671e5287bc01)

### data value counts()
![left](https://github.com/kavinesh8476/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118466561/59df1ad0-83a8-4b11-90f1-3a479ee5c062)

### data.head() for salary
![data head](https://github.com/kavinesh8476/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118466561/0ac307e2-690f-4a08-bc65-778317838cf8)

### x.head()
![xhead](https://github.com/kavinesh8476/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118466561/309ec9eb-1208-4d5c-971c-6654909bb43d)

### accuracy value
![accuracy](https://github.com/kavinesh8476/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118466561/66a5671d-6b01-4e5a-bbd2-2fc5b0722047)

### data prediction

![pred](https://github.com/kavinesh8476/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118466561/4b91ba0c-1e9f-4640-959d-d246bd7e68c2)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
