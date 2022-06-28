import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,auc,roc_curve,confusion_matrix,classification_report,roc_auc_score

heart_data = pd.read_csv("http://storage.googleapis.com/dimensionless/ML_with_Python/Chapter%205/heart.csv")
print(heart_data.shape)
print(heart_data.head(2))
#Check the information of data set
print(heart_data.info())
print(heart_data.describe())

#Creating Feature Matrix and Target Array
y = heart_data['target']
X = heart_data.drop(['target'],axis = 1)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=42,stratify=y)

#Apply Bagging Ensemble Method
model_bag = BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=3),
                              n_estimators=100,random_state=42,oob_score=True)
model_bag.fit(X_train,y_train)
print('OOB Score is: ',model_bag.oob_score_)
#Accuracy Score on Train data is
print('Accuracy Score on Train Data is: ',model_bag.score(X_train,y_train))
#Apply Predictions
y_pred_bag = model_bag.predict(X_test)
print('Confusion Matrix is: ',confusion_matrix(y_test,y_pred_bag))
sensitivity = (43/50)
print('Sensitivity of the model is: ',sensitivity)
specificity = (29/(29+12))
print('Specificity of the model is: ',specificity)
precision = (43/(43+12))
print('Precision of the model is: ',precision)
print('Accuracy Score on the Test Set is: ',accuracy_score(y_test,y_pred_bag))

#Apply Random Forest Classifier
model_rf = RandomForestClassifier(n_estimators=200,max_depth=5,max_features=12,
                                  oob_score=True,verbose=1,random_state=50)
model_rf.fit(X_train,y_train)
print('Score of the train set is: ',model_rf.score(X_train,y_train))
print('Feature Importances are: ',model_rf.feature_importances_)
data = pd.Series(data = model_rf.feature_importances_,index=X_train.columns)
data.sort_values(ascending=True,inplace=True)
data.plt.barh()
y_pred_test_rf = model_rf.predict(X_test)
print('Confusion Matrix is: ',confusion_matrix(y_test,y_pred_test_rf))
print('Accuracy Score is: ',accuracy_score(y_test,y_pred_test_rf))

#Perform Hyper parameter Tuning
parameters = {'max_features':[1,2,3,4,5,6,7,8,9],'max_depth':[1,2,3,4,5]}
tune_model = GridSearchCV(model_rf,parameters,cv=5,scoring='accuracy')
tune_model.fit(X_train,y_train)
print('Best Parameters are: ',tune_model.best_params_)
y_pred_test_cv = tune_model.predict(X_test)
print('Confusion Matrix on the Tuned model is: ',confusion_matrix(y_test,y_pred_test_cv))
print('Accuracy Score of the Tuned model is: ',accuracy_score(y_test,y_pred_test_cv))


