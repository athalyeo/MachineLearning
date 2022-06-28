import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import random
import seaborn as sns

Hitters = pd.read_csv("https://storage.googleapis.com/dimensionless/Analytics/Hitters.csv", index_col = 0)
print(Hitters.head(2))
Hitters = Hitters.dropna()
Hitters.Salary = np.log(Hitters.Salary)
Hitters.hist('Salary')

#Creating feature matrix and target array
X = Hitters.copy()
del X['Salary']
X = pd.get_dummies(X,columns = ['League', 'Division', 'NewLeague'],drop_first=True)
print(X.shape)
y = Hitters.Salary

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#Apply Ridge Regression
ridge_rg = linear_model.Ridge(alpha=0.1,normalize=True)
ridge_rg.fit(X_train,y_train)
print('Beta values for the model are: ',ridge_rg.coef_)
print('Score of the model is: ',ridge_rg.score(X_train,y_train))

#Apply CV on above model
np.random.seed(42) #42 is the randomly selected number
reg_cv = linear_model.RidgeCV(alphas = [0.01,0.001,0.1,1.0,10.0,100,1000,10000],cv = 5,normalize=True)
print('Best value of alpha is: ',reg_cv.alpha_)

#Now build the model with best value of alpha
ridge_cv = linear_model.Ridge(alpha=1,normalize=True)
ridge_cv.fit(X_train,y_train)
print('Score of Ridge CV model on train data is: ',ridge_cv.score(X_train,y_train))
#Apply the predictions
y_pred_rid = ridge_cv.predict(X_test)
SSE_cv = sum((y_test - y_pred_rid)**2)
print('SSE value is: ',SSE_cv)
SST = sum((y_test-np.mean(y_train))**2)
print('SST value is: ',SST)
r2_cv = (1 - (SSE_cv/SST))
print('R2 value is: ',r2_cv)
RMSE_cv = np.sqrt(SSE_cv/X_test.shape[0])
print('RMSE value is: ',RMSE_cv)

n_alphas = 200
alphas = np.logspace(-1,4,n_alphas)
coeffs = []
for a in alphas:
    ridge = linear_model.Ridge(alpha=a,normalize=True)
    ridge.fit(X_train,y_train)
    coeffs.append(ridge.coef_)

data = pd.DataFrame(coeffs,columns=X_train.columns,index = alphas)
data['alphas'] = data.index
print(data.columns)
y_var = data.columns.difference(['alphas'])
print(y_var)
#Now plot the representation
ax = plt.gca()
ax.set_xscale('log')
for i in range(0,19):
    ax = sns.lineplot(data = data,x='alphas',y = y_var[i])

#Implement Lasso Regularization Model
reg_lasso = linear_model.Lasso(alpha=0.001,normalize=True)
reg_lasso.fit(X_train,y_train)
print('Score of the model is: ',reg_lasso.score(X_train,y_train))
print('Beta values are: ',reg_lasso.coef_)

#Plotting the model coefficients
coef = pd.Series(reg_lasso.coef_,X_train.columns)
coef.sort_values(ascending=True,inplace=True)
coef.plot.barh()

#Apply CV for best Alpha value
np.random.seed(42)
reg_lasso_cv = linear_model.LassoCV(alphas=[0.0001,0.001,0.01,0.1,1],max_iter=10000,cv = 5,normalize=True)
reg_lasso_cv.fit(X_train,y_train)
print('Best alpha value is: ',reg_lasso_cv.alpha_)
#Apply the predictions now
y_pred_lasso = reg_lasso_cv.predict(X_test)
SSE_lasso = sum((y_test-y_pred_lasso)**2)
print('SSE value for Lassso Regularization is: ',SSE_lasso)
r2_lasso = (1-(SSE_lasso/SST))
print('R2 value for Lasso Regularization is: ',r2_lasso)
RMSE_lasso = np.sqrt(SSE_lasso/X_test.shape[0])
print('RMSE value for Lasso Regularization is: ',RMSE_lasso)

#Plotting Lasso Regularization for different alpha values
nalpha = 200
alphas = np.logspace(-4,-1,nalpha)
coef = []
for a in alphas:
    reg = linear_model.Lasso(alpha=a,max_iter=10000,normalize=True)
    reg_lasso_cv.fit(X_train,y_train)
    coef.append(reg_lasso_cv.coef_)
df_coef = pd.DataFrame(coef,index=alphas,columns = X_train.columns)
print(df_coef)

#Plot the figure
plt.figure(figsize=(10,10))
ax = plt.gca()
ax.plot(df_coef.index,df_coef.values)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('weights')
ax.get_ymajorticklabels()
plt.title('Lasso Coefficients a function of Regularization')
plt.axis('tight')
plt.legend(df_coef.columns)
plt.show()
