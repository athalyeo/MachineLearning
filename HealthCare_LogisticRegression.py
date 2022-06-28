import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, confusion_matrix, precision_score, recall_score, roc_curve,classification_report
import matplotlib.pyplot as plt
#import seaborn as sns #data visualisation

quality = pd.read_csv("https://storage.googleapis.com/dimensionless/Analytics/quality.csv")
print(quality.head(3))
print('Total Rows and Columns:',quality.shape)
print('Summary statistics of the data set:')
print(quality.describe())

#Analyze the Baseline model first
print(quality.PoorCare.value_counts())
print((98/131)*100)

#Creating a feature matrix
X = quality.iloc[:,0:13]
print(X.head(2))
y = quality['PoorCare']
print(y.head(2))

#Split the data into Train and Test sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=88,stratify=y)
print('Shape of X train is:',X_train.shape)
print('Shape of X test is:',X_test.shape)
print('Shape of Y train is:',y_train.shape)
print('Shape of Y test is:',y_test.shape)

#Display baseline model of y_train and y_test
print(y_train.value_counts())
print('Baseline accuracy of Train model is: ',(68/91))
print(y_test.value_counts())
print('Baseline accuracy of Test model is: ',30/40)

#Applying Logistic Regression Model
model = LogisticRegression()
model.fit(X_train[['Narcotics','OfficeVisits']],y_train)
print('Score of the train model is: ',model.score(X_train[['Narcotics','OfficeVisits']],y_train))
#Representation
#sns.lmplot("Narcotics","OfficeVisits",quality,fit_reg = False,hue = "PoorCare",height = 5,legent = True)
#plt.title("Model 1 Visualization")

#Predictions on the Train set
y_pred_train = model.predict(X_train[['Narcotics','OfficeVisits']])
print(y_pred_train)
#Display Confusion Matrix
print(confusion_matrix(y_train,y_pred_train))
#acc_score = ((66+9)/91)
#print('Accuracy  Score is: ',acc_score)
print('Accuracy Score is: ',accuracy_score(y_train,y_pred_train))

sensitivity = (9/(9+14)) # TP/TP+FN
print('Sensitivity is: ',sensitivity)
specificity = (66/(66+2)) #Specificity = TN/(TN+FP)
print('Specificity is: ',specificity)
prc = (9/11) #Precision = TP/TP+FP
print('Precision is: ',prc)

print('Precision SCore is: ',precision_score(y_train,y_pred_train))
print('Recall Score is:',recall_score(y_train,y_pred_train))

#Display summary
summary = classification_report(y_train,y_pred_train)
print('Summary Report is: ',summary)

#Predict the probabilities
pred_train_prob = model.predict_proba(X_train[['Narcotics','OfficeVisits']])
print('Probabilities are:',pred_train_prob) #Sum of the individual elements is 1

#ROC Curve
fpr , tpr, t = roc_curve(y_train,pred_train_prob[:,1],pos_label=1)
print('Threshold Value:',t)
print('FPR Values:',fpr) #FPR = (1-Specificity)
print('TPR Values:',tpr) #Sensitivity

#Plot the ROC  Curve
t[0] = 0.999999999
#%matplotlib inline
plt.scatter(fpr,tpr,c = t,s=100)
plt.colorbar(ticks = np.arrange(0,1,0.1))

#Applying Threshold on Predictions
pred_t_train = np.where(pred_train_prob[:,1] >= 0.3,1,0)
print(pred_t_train)
print('Confusion Matrix is: ',confusion_matrix(y_train,pred_t_train))
print('Accuracy Score is: ',accuracy_score(y_train,pred_t_train))
print('AUC value is: ',auc(fpr,tpr))

#Applying Predictions on Test Data
pred_test = model.predict(X_test[['Narcotics','OfficeVisits']])
print(pred_test)
print('Confusion Matrix on Test Data: ',confusion_matrix(y_test,pred_test))
print('Accuracy Score is:',accuracy_score(y_test,pred_test))
pred_test_prob = model.predict_proba(X_test[['Narcotics','OfficeVisits']])
print(pred_test_prob[:5])
pred_t_test = np.where(pred_train_prob[:,1] >= 0.3,1,0)
print(pred_t_test)
print('Confusion Matrix on Test Data: ',confusion_matrix(y_test,pred_t_test))
print('Accuracy Score is:',accuracy_score(y_test,pred_t_test))










