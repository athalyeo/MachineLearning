import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

Hitters = pd.read_csv("https://storage.googleapis.com/dimensionless/Analytics/Hitters.csv", index_col = 0)
print('Shape of the data set is: ',Hitters.shape)
print('Summary statistics of the data set is:',Hitters.describe())

Hitters = Hitters.dropna()

#Log normalize the target variable
Hitters.Salary = np.log(Hitters.Salary)

#Create Feature Matrix and Target Array
X = Hitters.copy()
del X['Salary']
X = pd.get_dummies(X,columns=['League','Division','NewLeague'],drop_first=True)
print('Shape of Independent Test Data is: ',X.shape)
y = Hitters.Salary

#Split Train and Test Data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#Apply PCA now
scalar = StandardScaler() #Apply Scaling
scalar.fit(X_train)

#Use transform method to  scale the data
scalar.transform(X_train)

#Transform the scaled data into dataframe
Scale_X = pd.DataFrame(scalar.transform(X_train),columns=X_train.columns)
#To check whether data is scaled or not check the mean and standard deviation values
#Mean for all the columns will come out to be 0 and std deviation will be 1
print('Mean values after scaling are: ',Scale_X.mean())
print('Standard deviation values after scaling are: ',Scale_X.std())

#Now create instance of PCA
pca_hitters = PCA()
#Now fot the scaled data
pca_hitters.fit(Scale_X)

#Phi values
print('Component or Phi values are: ',pca_hitters.components_)
#Transform component values into Dataframe
pca_loadings = pd.DataFrame(pca_hitters.components_,columns=X_train.columns)
#Transpose function - convert rows into columns and columns into rows
print(pca_loadings.T)

#Explained variance is the score of the variance of each principal component
#Explained variance ratio is explained variance divided by sum of variances

print('Variance Score of each column is: ',pca_hitters.explained_variance_)
print('Variance ratio score is: ',pca_hitters.explained_variance_ratio_)

PCscore = pd.DataFrame(pca_hitters.fit_transform(Scale_X),columns=pca_loadings.T.columns)
print(PCscore)

#Visualization
plt.plot(np.arange(1,20),np.cumsum(pca_hitters.explained_variance_ratio_),'-s',label = 'Cumulative')
plt.ylabel('Proportion of Variance explained')
plt.xlabel('Principal Component')
plt.xticks(np.arange(1,20))
plt.legend(loc=2)

#Apply the model now
pca_lm = LinearRegression()
#Fit the model on first 11 components as variace reduces after that
pca_lm.fit(PCscore.iloc[:,:11],y_train)
print('Beta values are: ',pca_lm.coef_)
print('Intercept values are: ',pca_lm.intercept_)

#Application on Test Data
Scale_Xtest = scalar.transform(X_test)
#Now transform it into Principal Components
pca_test = pca_hitters.transform(Scale_Xtest)
print('Shape after transformation of X test is: ',pca_test.shape)
#Apply the Predictions now:
y_pred_pca = pca_lm.predict(pca_test[:,:11])

#Performance Metrics
SSE_pca = sum((y_test-y_pred_pca)**2)
SST = sum((y_test - np.mean(y_train))**2)
r2_pca = (1-(SSE_pca/SST))
print('R2 score for PCA is: ',r2_pca)
