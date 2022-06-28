import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.feature_selection import RFE
import seaborn as sns

#Reading the data set
Htrain = pd.read_csv("https://storage.googleapis.com/dimensionless/ML_with_Python/Chapter%203%20Linear%20Regression/housing_train.csv")
print("Shape of the data set is ",Htrain.shape)
print("Checking whether data set has been imported successfully:")
print(Htrain.head(5))
#Median House Value is the dependent variable
print("Displaying the statistics of data set")
print(Htrain.describe())
#Checking for null values in each variable. If True it will return 1 and if False then 0 will be returned
Htrain.isnull().sum()

#Scatter plot between median income and Median house value
plt.scatter(Htrain[['median_income']],Htrain[['median_house_value']])
plt.xlabel('Median Income')
plt.ylabel('Median House Price')
plt.title('Corelation between Median Income and Median House Value')

print('Value of Corelation is:',Htrain['median_income'].corr(Htrain['median_house_value']))

#Apply the Linear Regresion algorithm now
lin_reg = LinearRegression()
#Train the model
lin_reg.fit(Htrain[['median_income']],Htrain['median_house_value'])

#Coeff or slope or m value is
print('Coefficient vallue is:',lin_reg.coef_)

#y intercept or beta or c value
print('Intercept value is: ',lin_reg.intercept_)

#R Square value is
print('Score of the Train model is: ',lin_reg.score(Htrain[['median_income']],Htrain['median_house_value']))

#Now predict the values on the Train set
y_pred_train = lin_reg.predict(Htrain[['median_income']])
#Print the first 5 predicted values
print(y_pred_train[:5])

#Calculate Root Mean Squarred Value
lin_rmse = np.sqrt(mean_squared_error(Htrain['median_house_value'],y_pred_train))
print("RMSE value is: ",lin_rmse)

#Draw the Scatter plot of Linear Regression prediction
plt.scatter(Htrain[['median_income']],Htrain[['median_house_value']])
plt.plot(Htrain[['median_income']],y_pred_train,'r')
plt.axis([-2,6,0,550000])
plt.xlabel('Median Income')
plt.ylabel('Median House Price')
plt.title('Linear Regression on Median Income and Median House Value')

# Find out the best FIT using Gradient Descent Regression
gdreg = SGDRegressor()
gdreg.fit(Htrain[['median_income']],Htrain['median_house_value'])
print('Coefficient value is ',gdreg.coef_)
print('Intercept value is: ',gdreg.intercept_)
print('Score of the model is:')
print(gdreg.score(Htrain[['median_income']],Htrain['median_house_value']))
y_pred_gd = gdreg.predict(Htrain['median_income'])
#Display the Scatter Plot
plt.scatter(Htrain[['median_income']],Htrain[['median_house_value']])
plt.plot(Htrain[['median_income']],y_pred_gd,'r')
plt.axis([-2,6,0,550000])
plt.xlabel('Median Income')
plt.ylabel('Median House Price')
plt.title('Best Fit using Gradient Descent')

#Immplementing Multiple Linear Regressison
model_all = LinearRegression()
#Dropping the target variable from Independent variable set
X_train = Htrain.drop('median_house_value',axis = 1)
y_train = Htrain['median_house_value']
model_all.fit(X_train,y_train)
print('Coefficient value is: ',model_all.coef_)
print('Intercept value is: ',model_all.intercept_)
print('Score of the model is: ',model_all.score(X_train,y_train))
#Predict on the Training set
y_pred_all = model_all.predict(X_train)
rmse_all = np.sqrt(mean_squared_error(y_train,y_pred_all))
print("RMSE value is: ",rmse_all)

#Display Corelation table to understand important features
print(Htrain.corr())

#Check model with Longitude variable
model_one = LinearRegression()
model_one.fit(X_train[['longitude']],y_train)
print('Coefficient value is: ',model_one.coef_)
print('Intercept value is: ',model_one.intercept_)
print('Score of the model is: ',model_one.score(X_train[['longitude']],y_train))

#Check model with Longitude and Latitude variables
model_two = LinearRegression()
model_two.fit(X_train[['longitude','latitude']],y_train)
print('Coefficient value is: ',model_two.coef_)
print('Intercept value is: ',model_two.intercept_)
print('Score of the model is: ',model_two.score(X_train[['longitude','latitude']],y_train))

plt.scatter(x=Htrain['longitude'],y=Htrain['latitude'],c=Htrain['median_house_value'])

#Implementing feature Selection :
#Select 6 best features based on f_regression metric
select_feature = SelectKBest(f_regression,6)
print(select_feature.scores_)
print(select_feature.pvalues_)
selected_features_df = pd.DataFrame({'Feature':list(X_train.columns),
                                     'P_values':select_feature.pvalues_})
print(selected_features_df.sort_values(by='P_values',ascending=True))

#Use the features decide from above code base for modelling
X_train_new = select_feature.transform(X_train)
print(X_train_new.shape)
model_kbest = model_all.fit(X_train_new,y_train)
print(model_kbest.coef_)
print("R2 score of the model is: ",model_kbest.score(X_train_new,y_train))
y_pred_kbest = model_kbest.predict(X_train_new)
lin_rmse_new = np.sqrt(mean_squared_error(y_train,y_pred_kbest))
print('RMSE value is: ',lin_rmse_new)

#Implement Recursive Feature elimination method
#Step 1 means eliminating one feature at each iteration
rfe = RFE(estimator=model_all,step = 1)
rfe.fit(X_train,y_train)
print(X_train.columns)
print(rfe.ranking_)
# Now select the features with RFE
selected_rfe_features = pd.DataFrame({'Feature':list(X_train.columns),
                                      'Ranking':rfe.ranking_})
print(selected_rfe_features.sort_values(by='Ranking'))

X_train_rfe=rfe.transform(X_train)
print(X_train_rfe.shape)
model_rfe = model_all.fit(X_train_rfe,y_train)
print('Intercept value is: ',model_rfe.intercept_)
print('Coefficient value is: ',model_rfe.coef_)
print('Score of the model is: ',model_rfe.score(X_train_rfe,y_train))
y_pred_rfe = model_rfe.predict(X_train_rfe)
lin_rmse_rfe = np.sqrt(mean_squared_error(y_train,y_pred_rfe))
print('RMSE value is: ',lin_rmse_rfe)

#Applying prediction on Test Data
Htest = pd.read_csv("https://storage.googleapis.com/dimensionless/ML_with_Python/Chapter%203%20Linear%20Regression/housing_test.csv")
X_test = Htest.drop('median_house_value',axis = 1)
y_test = Htest['median_house_value']
model_test = LinearRegression()
#Fitting the model on Train set
model_test.fit(X_train,y_train)
y_pred_test = model_test.predict(X_test)
print(y_pred_test)
rmse_test= np.sqrt(mean_squared_error(y_test,y_pred_test))
print('Test RMSE value is: ',rmse_test)

#Calculating Residual Sum of errors
SSE = np.sum((y_pred_test - y_test)**2)
#Calculating Total sum of errors
SST = np.sum((y_test-np.mean(y_train))**2)
#Calculate value of R2 using SST and SSE
r2_test = 1 - (SSE/SST)

print('Test SSE value is: ',SSE)
print('Test SST value is: ',SST)
print('Test R2 value is : ',r2_test)

#Plot the Heatmap
cor = Htrain.corr()
sns.heatmap(cor,annot = True)
