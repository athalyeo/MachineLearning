import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

Hitters = pd.read_csv("https://storage.googleapis.com/dimensionless/Analytics/Hitters.csv", index_col = 0)
print(Hitters.head(2))
print('Shape of the data set is: ',Hitters.shape)
print('Summary statistics of the data set is:',Hitters.describe())

#Data Preprocessing
Hitters = Hitters.dropna()
print('Updated Shape is:',Hitters.shape)
print('Columns are:',Hitters.columns)

#Histogram representation of Salary is heavily positively skewed. So convert to log scale
Hitters.Salary = np.log(Hitters.Salary)
#For regression models, target variables with normal distribution provides a better fit compared to target variable
#with skewed distribution
Hitters = pd.get_dummies(Hitters,columns = ['League','Division','NewLeague'],drop_first=True)
print(Hitters.head(2))

#Creating Feature Matrix and Target Arrays
X = Hitters.copy()
print(X.columns)
del X['Salary']
print(X.head(2))
y = Hitters.Salary
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print('Shape of X train',X_train.shape)
print('Shape of X test',X_test.shape)
print('Shape of y train',y_train.shape)
print('Shape of y test',y_test.shape)

#Implement Bagging Regressor
#n_estomatiire = 100 means we will build 100 decision trees
#OOB Score = suppose out 210, 140 samples are taken as train set and 70 samples as test set
#then model is trained on 140 samples and tested on 70 samples. Then oob score is calulated for this model
#Overall once processing is done for all samples then average of oob score is calulated which is given by model_bag.oob_score_

#OOB score is just like R square for regression and accuracy for classification
model_bag = BaggingRegressor(base_estimator=DecisionTreeRegressor(max_depth=3),n_estimators=100,oob_score=True,
                             random_state=42,max_features=19)
model_bag.fit(X_train,y_train)
print('OOB score is: ',model_bag.oob_score_)
y_pred_bag = model_bag.predict(X_test)
rmse_bag = np.sqrt(mean_squared_error(y_test,y_pred_bag))
print('RMSE score is: ',rmse_bag)

#Implement Random Forest Regressor
model_rf = RandomForestRegressor(n_estimators=200,max_features=12,random_state=42,oob_score=True,
                                 verbose=2,max_depth=6)
model_rf.fit(X_train,y_train)
#OOB score - For Unseen data
print('OOB score is: ',model_rf.oob_score_)
#Score of the train data - that is on seen data
print('R2 Score is: ',model_rf.score(X_train,y_train))
print('Feature Importances: ',model_rf.feature_importances_)
data = pd.Series(model_rf.feature_importances_,index=X_train.columns)
data.sort_values(ascending=True,inplace=True)
data.plot.barh()

#apply predictions now
y_pred_rf = model_rf.predict(X_test)
SSE = mean_squared_error(y_test,y_pred_rf)*y_test.shape[0]
print('SSE value is: ',SSE)
SST = np.sum((y_test -  np.mean(y_train))**2)
print('SST value is: ',SST)
R2 = (1 - (SSE/SST))
print('R2 value is: ',R2)
rmse_rf = np.sqrt(mean_squared_error(y_test,y_pred_rf))
print('RMSE value for Random Forest Regression is: ',rmse_rf)

#Apply Cross Validation on above model
parameters = {'max_depth':np.arange(4,7),'max_features':np.arange(10,15)}
tune_model = GridSearchCV(model_rf,parameters,cv=5)
tune_model.fit(X_train,y_train)
print('Best parameter is: ',tune_model.best_params_)

#Apply prediction on test set
y_pred_cv = tune_model.predict(X_test)
rmse_cv= np.sqrt(mean_squared_error(y_test,y_pred_cv))
print('RMSE value for the tuned model is: ',rmse_cv)

#Main difference between Random Forest and XG Regressor is that in Random forest trees are built parallely
#however in XGBoost treees are  build one after another. XGboost continuously tries to minimize the error
#using gradient decent method and lost function used is squared error

#Implement boosting
model_xgb = xgb.XGBRegressor(objective = 'reg:squarederror',n_estimators = 500, max_depth = 3)
model_xgb.fit(X_train,y_train)
print('R2 value of the train set is: ',model_xgb.score(X_train,y_train))
y_pred_xgb = model_xgb.predict(X_test)
SSE = sum((y_test - y_pred_xgb)**2)
print('SSE value is: ',SSE)
SST = sum((y_test - np.mean(y_train))**2)
print('SST value is: ',SST)
r2_xgb = 1 - (SSE/SST)
print('R2 for XGB model is: ',r2_xgb)
#R2 value on test set can also be calculated by invoking score method and passing X_test and y_test to it
rmse_xgb = np.sqrt(mean_squared_error(y_test,y_pred_xgb))
print('RMSE value for XGB model is: ',rmse_xgb)

#Now perform Hyper parameter Tuning on XGB model
param_grid = {'max_depth':np.arange(1,4),'learning_rate':[0.1,0.01,0.001]}
#If cv parameter is not set in GridSearchCv method then default value of cv = 5 is taken
tune_model_xgb = GridSearchCV(model_xgb,param_grid,cv = 5)
tune_model_xgb.fit(X_train,y_train)
print('Best Parameter for the XGB tuned model is: ',tune_model_xgb.best_params_)
print('Average best score for the XGB Tuned model is: ',tune_model_xgb.best_score_)

#Apply Cross Validations Now
model_xgb_cv = xgb.XGBRegressor(objective='reg:linear',n_estimators=1000,max_depth=2,learning_rate=0.01)
model_xgb_cv.fit(X_train,y_train)
#Apply predictions now
y_pred_xgb_cv = model_xgb_cv.predict(X_test)
SSE = sum((y_test - y_pred_xgb_cv)**2)
print('SSE value for CV model on XGB is: ',SSE)
SST = sum((y_test - np.mean(y_train))**2)
print('SST value for CV model on XGB is: ',SST)
r2 = (1 - (SSE/SST))
print('R2 value for CV model on XGB is: ',r2)
rmse_cv_xgb = np.sqrt(mean_squared_error(y_test,y_pred_xgb_cv))
print('RMSE value for CV model on XGB is: ',rmse_cv_xgb)


