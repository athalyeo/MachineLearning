import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('https://storage.googleapis.com/dimensionless/Analytics/USArrests.csv', index_col=0)
print("Before scaling head is ",df.head())
print("Mean before scaling is ",df.mean())
print("Variance before scaling is ",df.var())

#As Assault has highest variance, output will be driven by it
# So there is need to standardize the variables.We never apply PCA without standardizing the data
print(scale(df))

#Now create a dataframe as we have to pass it to fit method.
X = pd.DataFrame(scale(df),index=df.index,columns=df.columns)
print(X.head(4))
print('Standard deviation after scaling is: ',X.std())
print('Mean values after scaling are: ',X.mean())
print('Variance after scaling is: ',X.var())

#After standardizing, output will not be driven Assault, it will be driven by linear combination of variables
pca = PCA()
#As it is unsupervised, we dont require y data
pca.fit(X)
#Number of principal components
print(pca.n_components_)
#Component values
print(pca.components_)

#Transform the components into DataFrames
pca_loadings = pd.DataFrame(pca.components_,index=['PC1','PC2','PC3','PC4'],columns=['Murder','Assault','UrbanPop','Rape'])
print(pca_loadings.T)
PCscore = pd.DataFrame(pca.transform(X),index=df.index, columns = ['PC1','PC2','PC3','PC4'])
print(PCscore)

#Variance Explained by each principal component
print('Explained variance of each principal component is: ',pca.explained_variance_)
print('Explained variance ratio is: ',pca.explained_variance_ratio_)
print('Mean values: ',pca.mean_)

#Using the Scree plot we can decide with how many principal components we can go ahead
#For this example we can go with 3 as for the first two there is lot of variance.4th component does
#not have lot of variance so the varaince is covered in first 3 components
plt.figure(figsize=(7,5))
plt.plot([1,2,3,4],pca.explained_variance_ratio_,'-o',label = 'Individual Component')
plt.plot([1,2,3,4],np.cumsum(pca.explained_variance_ratio_),'-s',label = 'Cumulative')
plt.ylabel('Proportion of Variance explained')
plt.xlabel('Principal Component')
plt.xlim(0.75,4.25)
plt.ylim(0,1.05)
plt.xticks([1,2,3,4])
plt.legend(loc=2)
plt.show()

