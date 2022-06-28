import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,auc, roc_curve, confusion_matrix, classification_report,roc_auc_score
from sklearn import tree

heart_data = pd.read_csv("http://storage.googleapis.com/dimensionless/ML_with_Python/Chapter%205/heart.csv")
print(heart_data.shape)
print(heart_data.head(2))
#Check the information of data set
print(heart_data.info())
print(heart_data.describe())

#Analyze the data by displaying Histograms
heart_data.hist(figsize=(10,10))

#Check Column names
print(heart_data.columns)

#Check the value counts
print(heart_data['target'].value_counts())
print('Baseline accuracy is:',((165/303)*100))

#Creating Feature Matrix and Target Array
y = heart_data['target']
X = heart_data.drop(['target'],axis = 1)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=42,stratify=y)

#Now build the model
hd_model = DecisionTreeClassifier(max_depth=4,random_state=100)
hd_model.fit(X_train,y_train)
plt.figure(figsize=(25,10))
tree.plot_tree(hd_model,feature_names=heart_data.columns,filled=True,label='all',fontsize=10)
plt.show()
print('Score of the model is',hd_model.score(X_train,y_train))
y_pred_test = hd_model.predict(X_test)
print(y_pred_test)
print('Confusion Matrix is:',confusion_matrix(y_test,y_pred_test))
print('Accuracy Score is:',accuracy_score(y_test,y_pred_test))
sensitivity = (42/(42+8))
print('Sensitivity is:',sensitivity)
specificity = (28/(28+13))
print('Specificity is:',specificity)

#Plotting ROC Curve and Finding AUC
pred_probab = hd_model.predict_proba(X_test)
print(pred_probab[:,1])
fpr,tpr,t = roc_curve(y_test,pred_probab[:,1],pos_label=1)
print(fpr)
print(tpr)
print(t)
print('Area Under Curve is:',auc(fpr,tpr))

#Prune the model
model1= DecisionTreeClassifier(random_state=100)
model1.fit(X_train,y_train)
parameters = {'max_depth':[1,2,3,4,5,6,7,8]}
grid = GridSearchCV(model1,parameters,cv = 10,scoring='accuracy')
grid.fit(X_train,y_train)
print('Best parameters are: ',grid.best_params_)
#Average score of the best parameter value. Average of 80 models build with above cv=10 and max depth
print('Score of the pruned model is: ',grid. best_score_)

#Now build the model with best depth value
model_prune = DecisionTreeClassifier(max_depth=3,random_state=42)
model_prune.fit(X_train,y_train)
print('Score of the pruned model is: ',model_prune.score(X_train,y_train))

#Feature Importances
print('Feature Importances: ',model_prune.feature_importances_)
data = pd.Series(data=model_prune.feature_importances_,index=X.columns)
data.sort_values(ascending=True,inplace=True)
data.plot.barh()

#Apply the predictions on test data
pred_test = model_prune.predict(X_test)
print(pred_test)
print('Confusion Matrix is: ',confusion_matrix(y_test,y_pred_test))
print('Accuracy Score is: ',accuracy_score(y_test,pred_test))
sensitivity = (40/(40+10))
print('Sensitivity of Test Data is: ',sensitivity)
specificity = (28/(28+13))
print('Specificity of test data is: ',specificity)
precision = (40/(40+13))
print('Precision value is: ',precision)

#Representation
plt.figure(figsize=(15,10))
tree.plot_tree(model_prune,fontsize=10,filled=True,feature_names=X_train.columns)
plt.show()