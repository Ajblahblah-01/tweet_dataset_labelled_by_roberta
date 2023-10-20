import numpy as np
import pandas as pd
from sklearn import metrics 
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from feature import generate_data_set
from sklearn.model_selection import cross_val_score
# Gradient Boosting Classifier Model
from sklearn.ensemble import GradientBoostingClassifier
#Random Forest Classifier Model
from sklearn.ensemble import RandomForestClassifier
#Decision Tree Classifier Model
from sklearn.tree import DecisionTreeClassifier   
#Support Vector Classifier Model 
from sklearn.svm import SVC
#Logistic Regression Model
from sklearn.linear_model import LogisticRegression  
#K-NN Classifier Model
from sklearn.neighbors import KNeighborsClassifier  
#Ada Boost Classifier Model
from sklearn.ensemble import AdaBoostClassifier
#XGBoost Classifier Model
from xgboost import XGBClassifier

data = pd.read_csv("phishing.csv")

#droping index column
data = data.drop(['Index'],axis = 1)

# Splitting the dataset into dependant and independant fetature
X = data.drop(["class"],axis =1)
y = data["class"]

kf = KFold(n_splits=5, random_state=42, shuffle=True)
for train_index, val_index in kf.split(X):
	X_train, X_val = X.iloc[train_index], X.iloc[val_index]
	y_train, y_val = y.iloc[train_index], y.iloc[val_index]

#  GBC Model
gbc = GradientBoostingClassifier(max_depth=4, learning_rate=0.7)
gbc.fit(X_train,y_train)
print("Accuracy Score for Gradient Boosting Classifier: \n", "{:.2f}".format(accuracy_score(y_val, gbc.predict(X_val))*100))

#RFC Model
rfc= RandomForestClassifier(n_estimators= 10, criterion="entropy")  
rfc.fit(X_train, y_train)  
print("Accuracy Score for Random Forest Classifier: \n", "{:.2f}".format(accuracy_score(y_val, rfc.predict(X_val))*100))

#DTC Model
dtc = DecisionTreeClassifier(criterion='entropy', random_state=0)  
dtc.fit(X_train, y_train)
print("Accuracy Score for Decision Tree Classifier: \n", "{:.2f}".format(accuracy_score(y_val, dtc.predict(X_val))*100))

#SVC Model
svc = SVC(kernel='linear', random_state=0)  
svc.fit(X_train, y_train)
print("Accuracy Score for Support Vector Classifier: \n", "{:.2f}".format(accuracy_score(y_val, svc.predict(X_val))*100))

#LR Model
lr = LogisticRegression(random_state=0)  
lr.fit(X_train, y_train)  
print("Accuracy Score for Logistic Regression: \n", "{:.2f}".format(accuracy_score(y_val, lr.predict(X_val))*100))

#KNN Model
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )  
knn.fit(X_train, y_train)  
print("Accuracy Score for KNN Classifier: \n", "{:.2f}".format(accuracy_score(y_val, knn.predict(X_val))*100))

#Ada Boost Model
ada = AdaBoostClassifier(random_state=0)
ada.fit(X_train, y_train)
print("Accuracy Score for Ada Boost Classifier: \n", "{:.2f}".format(accuracy_score(y_val, ada.predict(X_val))*100))

#XGBoost Model
xgb = XGBClassifier(random_state=0)
xgb.fit(X_train, y_train)
print("Accuracy Score for XGBoost Classifier: \n", "{:.2f}".format(accuracy_score(y_val, xgb.predict(X_val))*100))

