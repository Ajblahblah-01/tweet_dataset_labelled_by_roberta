#importing required libraries

from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn import metrics 
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold
from feature import generate_data_set
# XG Boost Classifier Model
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

# instantiate the model
xgb = XGBClassifier(random_state=0)
xgb.fit(X_train, y_train)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html", xx= -1)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":

        url = request.form["url"]
        x = np.array(generate_data_set(url)).reshape(1,30) 
        y_pred =xgb.predict(x)[0]
        #1 is safe       
        #-1 is unsafe
        y_pro_phishing = xgb.predict_proba(x)[0,0]
        y_pro_non_phishing = xgb.predict_proba(x)[0,1]
        # if(y_pred ==1 ):
        pred = "It is {0:.2f} % safe to go ".format(y_pro_phishing*100)
        return render_template('index.html',xx =round(y_pro_non_phishing,2),url=url )
        # else:
        #     pred = "It is {0:.2f} % unsafe to go ".format(y_pro_non_phishing*100)
        #     return render_template('index.html',x =y_pro_non_phishing,url=url )
    return render_template("index.html", xx =-1)


if __name__ == "__main__":
    app.run(debug=True)