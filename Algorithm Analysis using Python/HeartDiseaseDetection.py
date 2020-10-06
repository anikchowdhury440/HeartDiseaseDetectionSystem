# Data Preprocessing
# importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Read dataset
dataset = pd.read_csv('dataset1.csv')
#read csv file
#Dataset contains following features:
#age — age in years
#sex — (1 = male; 0 = female)
#cp — chest pain type
#trestbps — resting blood pressure (in mm Hg on admission to the hospital)
#chol — serum cholestoral in mg/dl
#fbs — (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
#restecg — resting electrocardiographic results
#thalach — maximum heart rate achieved
#exang — exercise induced angina (1 = yes; 0 = no)
#oldpeak — ST depression induced by exercise relative to rest
#slope — the slope of the peak exercise ST segment
#ca — number of major vessels (0–3) colored by flourosopy
#thal — 3 = normal; 6 = fixed defect; 7 = reversable defect
#target — have disease or not (1=yes, 0=no)

dataset.info()

# Data Visualization

dataset.hist(figsize=(26,16))

# Heart disease frequency by age
plt.figure(figsize=(15, 15))
sns.countplot(x='age', hue='target', data=dataset, palette=['green', 'red'])
plt.legend(["No Disease", "Have Disease"])
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Maximum heart rate vs age
plt.scatter(x=dataset.age[dataset.target==1], y=dataset.thalach[(dataset.target==1)], c="red")
plt.scatter(x=dataset.age[dataset.target==0], y=dataset.thalach[(dataset.target==0)], c = 'blue')
plt.legend(["Disease", "Not Disease"])
plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate")
plt.title("Maximum Heart Rate vs Age")
plt.show()

# Frequency of Chest pain type
pd.crosstab(dataset.cp,dataset.target).plot(kind="bar",figsize=(15,6),color=['red','blue' ])
plt.title('Heart Disease Frequency for Chest Pain Type')
plt.xlabel('Chest Pain Type')
plt.xticks(rotation=0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.show()

corr = dataset.corr()
print(corr)
# Compute pairwise correlation of columns, excluding NA/null values.
# covariance(X, Y) = (sum (x - mean(X)) * (y - mean(Y)) ) 
# correlation coefficient = covariance(X, Y) / (stdv(X) * stdv(Y))

sns.heatmap(corr, xticklabels = corr.columns, yticklabels = corr.columns)

# Splittig data in training and testing and standardisation of data
# Find independent variables
X = dataset.iloc[:,:-1].values

# Find dependent variables
y = dataset.iloc[:,13].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X1_train = sc_X.fit_transform(X_train)
X1_test = sc_X.transform(X_test)


# KNN Algorithm

from sklearn.neighbors import KNeighborsClassifier
error_rate = []
for k in range(1,21):
    knn_classifier = KNeighborsClassifier(n_neighbors = k,metric = 'minkowski', p=2, weights ='uniform')
    # n_neighbors - Number of neighbors to use, weights - ‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.
    # metric - the distance metric to use for the tree. The default metric is minkowski, and with p=2 is equivalent to the standard Euclidean metric. 
    knn_classifier.fit(X1_train, y_train)
    knn_y_pred=knn_classifier.predict(X1_test)
    error_rate.append(np.mean(knn_y_pred !=y_test))

#predicting error for every k value   
plt.figure(figsize=(15,9))
plt.plot(range(1,21),error_rate,color='blue',linestyle='dashed',marker=
'o',markerfacecolor='red',markersize=8)
plt.title('Error Rate Vs K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

min_error = min(error_rate)
optimal_k = error_rate.index(min(error_rate))+1

knn_classifier =KNeighborsClassifier(n_neighbors=optimal_k,metric = 'minkowski', p=2, weights ='uniform')
knn_classifier.fit(X1_train,y_train)

knn_y_pred =knn_classifier.predict(X1_test)

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
knn_cm = confusion_matrix(y_test,knn_y_pred)

# True Positive (TP) : Observation is positive, and is predicted to be positive. (bottom right)
# False Negative (FN) : Observation is positive, but is predicted negative. (botttom left)
# True Negative (TN) : Observation is negative, and is predicted to be negative.(top left)
# False Positive (FP) : Observation is negative, but is predicted positive. (top right)

accuracy_knn = knn_classifier.score(X1_test, y_test)


# SVM Algorithm
from sklearn.svm import SVC
svm_accuracy = []
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for i in range(len(kernels)):
    svm_classifier = SVC(kernel = kernels[i])
    # kernel - Specifies the kernel type to be used in the algorithm. 
    svm_classifier.fit(X1_train,y_train)
    svm_y_pred = svm_classifier.predict(X1_test)
    svm_accuracy.append(svm_classifier.score(X1_test, y_test))
    
plt.figure(figsize=(15,9))
plt.bar(kernels, svm_accuracy, align = 'center')
for i in range(len(kernels)):
    plt.text(i, svm_accuracy[i],svm_accuracy[i])
plt.xlabel('Kernels')
plt.ylabel('Scores')
plt.title('Support Vector Classifier scores for different kernels')

max_accuracy = max(svm_accuracy)
optimal_kernel = kernels[svm_accuracy.index(max(svm_accuracy))]

#Fittiing the classifier to the Training Set
svm_classifier = SVC(kernel = optimal_kernel)
svm_classifier.fit(X1_train,y_train)

#Predicitng the Test Set results
svm_y_pred = svm_classifier.predict(X1_test)

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
svm_cm = confusion_matrix(y_test,svm_y_pred)

accuracy_svm = svm_classifier.score(X1_test, y_test)

#Testing realtime data

patient_age = int(input("Enter Patient Age (in years):  "))
patient_sex = int(input("Enter Patient Gender(1 = male; 0 = female):  "))
patient_cp = int(input("Enter Patient Chest Pain Type (0 = typical angina, 1 = atypical angina, 2 = non — anginal pain, 3 = asymptotic):  "))
patient_trestbps = int(input("Enter Patient Blood Pressure (in mmHg )(normal:- 80/140):  "))
patient_chol = int(input("Enter Patient Cholestrol (in mg/dl ) (normal:- around 200):  "))
patient_fbs = int(input("Enter Patient Fasting blood sugar (fasting blood sugar > 120 mg/dl 1 = true; 0 = false ):  "))
patient_restecg = int(input("Enter Patient resting electrocardiographic results (0 = normal, 1 = having ST-T wave abnormality, 2 = left ventricular hyperthrophy):  "))
patient_thalach = int(input("Enter Patient max heart rate achieved:  "))
patient_exang = int(input("Enter Patient Exercise induced angina(1 = yes, 0 = no):  "))
patient_oldpeak = float(input("Enter ST depression induced by exercise relative to rest ( 0 - 4):  "))
patient_slope = int(input("Enter Peak exercise ST segment  (1 = upsloping, 2 = flat,3 = downsloping):  "))
patient_ca = int(input("Enter Number of major vessels (0–3) colored by flourosopy:  "))
patient_thal = int(input("Enter patient thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect):  "))

patient_data = {'age':  [patient_age],
                 'sex':  [patient_sex],
                 'cp':  [patient_cp],
                 'trestbps':  [patient_trestbps],
                 'chol':  [patient_chol],
                 'fbs':  [patient_fbs],
                 'restecg':  [patient_restecg],
                 'thalach':  [patient_thalach],
                 'exang':  [patient_exang],
                 'oldpeak':  [patient_oldpeak],
                 'slope':  [patient_slope],
                 'ca':  [patient_ca],
                 'thal':  [patient_thal],}

patient_dataset = pd.DataFrame (patient_data, columns = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'])

patient_X = patient_dataset.iloc[:,:].values

patient_data1 = sc_X.transform(patient_X)

knn_patient_target = knn_classifier.predict(patient_data1) 
svm_patient_target = svm_classifier.predict(patient_data1)

print("Prediction by knn algorithm: ", knn_patient_target)
print("Prediction by svm algorithm: ", svm_patient_target)


