import matplotlib
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn import tree

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

#Build the model
model2_var = DecisionTreeRegressor(max_depth=2)
model2_var.fit(X_train[["Years","Hits"]],y_train)
plt.figure(figsize=(9,8))
tree.plot_tree(model2_var,feature_names=['Years','Hits'],filled=True,label='all')

#Build tree with all the variables
model_all = DecisionTreeRegressor(max_depth=4)
model_all.fit(X_train,y_train)
plt.figure(figsize=(20,15))
tree.plot_tree(model_all,feature_names=X_train.columns,filled=True,label='all',fontsize=10)
print('Score of the model is:',model_all.score(X_train,y_train))
y_pred_test = model_all.predict(X_test)
SSE_m1 = np.sum((y_pred_test - y_test)**2)
print('Sum of Square errors is: ',SSE_m1)
SST_m1 =  np.sum((y_test-np.mean(y_train))**2)
print('Total sum of squares:',SST_m1)
R2_m1 = (1 - (SSE_m1/SST_m1))
print('R2 value of model1 is :',R2_m1)
rmse_m1 =  np.sqrt(mean_squared_error(y_test,y_pred_test))
print('RMSE value is:',rmse_m1)

#Calculating % RMSE value
#RMPSE is better metric than RMSE as it gives us Percentage error
rmpse_m1 = (np.sqrt(np.nanmean(np.square(((y_test - y_pred_test) / y_test))))*100)
print('RMSE Percentage value is: ',rmpse_m1)

#Pruning the Tree
parameters = {'max_depth':[1,2,3,4,5,6]}
np.arrange(1,9)
grid = GridSearchCV(model_all,parameters,cv = 5, scoring='r2')
grid.fit(X_train,y_train)
print('Best parameters are:',grid.best_params_)
print('Score associated with these hyper parametr values is:',grid.best_score_)

#Now build the model with ideal max depth
model_prune = DecisionTreeRegressor(max_depth=3)
model_prune.fit(X_train,y_train)
print('Score of train  model is:',model_prune.score(X_train,y_train))
pred_test = model_prune.predict(X_test)
SSE_Prune = np.sum((pred_test - y_test)**2)
print('Sum of squares error on Pruned model is: ',SSE_Prune)
SST_Prune = np.sum((y_test - np.mean(y_train))**2)
print('Total Sum of Errors is: ',SST_Prune)
R2_Prune = (1 - (SSE_Prune/SST_Prune))
print('R2 score of the pruned model is: ',R2_Prune)

#Feature Importances
print('Feature importances are:',model_prune.feature_importances_)
data = pd.Series(model_prune.feature_importances_,index=X_train.columns)
print(data.to_frame().reset_index())
data.sort_values(ascending=True,inplace=True)
print(data.sort_values(ascending=False).to_frame())
data.plot.barh()
plt.figure(figsize=(15,10))
tree.plot_tree(model_prune,feature_names=X_train.columns,filled=True,label='all')

#Get the original values beacuse we had taken log earlier
dav = np.exp(pred_test)
print(dav)

