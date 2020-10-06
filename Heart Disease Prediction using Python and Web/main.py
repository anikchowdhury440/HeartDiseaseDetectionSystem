from flask import Flask,render_template,request
app = Flask(__name__)
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/result", methods = ["GET","POST"])
def result():
    if request.method == "POST":
        dataset = pd.read_csv('dataset1.csv')
        X = dataset.iloc[:,:-1].values
        y = dataset.iloc[:,13].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
        sc_X = StandardScaler()
        X1_train = sc_X.fit_transform(X_train)
        X1_test = sc_X.transform(X_test)
        knn_classifier =KNeighborsClassifier(n_neighbors=7,metric = 'minkowski', p=2, weights ='uniform')
        knn_classifier.fit(X1_train,y_train)
        svm_classifier = SVC(kernel = 'poly', random_state = 42,gamma='auto')
        svm_classifier.fit(X1_train,y_train)
        myDict = request.form
        name = myDict['name']
        age = int(myDict['age'])
        sex = int(myDict['sex'])
        cp = int(myDict['cp'])
        trestbps = int(myDict['trestbps'])
        chol = int(myDict['chol'])
        fbs = int(myDict['fbs'])
        restecg = int(myDict['restecg'])
        thalach = int(myDict['thalach'])
        exang = int(myDict['exang'])
        oldpeak = float(myDict['oldpeak'])
        slope = int(myDict['slope'])
        ca = int(myDict['ca'])
        thal = int(myDict['thal'])
        patient_data = {'age':  [age],
                 'sex':  [sex],
                 'cp':  [cp],
                 'trestbps':  [trestbps],
                 'chol':  [chol],
                 'fbs':  [fbs],
                 'restecg':  [restecg],
                 'thalach':  [thalach],
                 'exang':  [exang],
                 'oldpeak':  [oldpeak],
                 'slope':  [slope],
                 'ca':  [ca],
                 'thal':  [thal],}
        patient_dataset = pd.DataFrame (patient_data, columns =['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'])
        patient_X = patient_dataset.iloc[:,:].values
        patient_data1 = sc_X.transform(patient_X)
        knn_pred = knn_classifier.predict(patient_data1)
        svm_pred = svm_classifier.predict(patient_data1)
        print(knn_pred)
        print(svm_pred)
        if knn_pred == 1 and svm_pred == 1:
            prediction = 1
        elif knn_pred == 0 and svm_pred == 0:
            prediction = 0
        else:
            prediction = 2
    
    return render_template("result.html",result = prediction,name = name,sex= sex,cp = cp,restecg = restecg,exang = exang,slope = slope,thal = thal,trestbps = trestbps,age = age,chol = chol,thalach = thalach,oldpeak = oldpeak,ca = ca,fbs = fbs)

if __name__ == "__main__":
    app.run(debug=True)