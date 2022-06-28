import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans as kmeans,AgglomerativeClustering as hclust
from scipy.cluster.hierarchy import dendrogram,ward
import matplotlib.pyplot as plt

movies=pd.read_csv("http://files.grouplens.org/datasets/movielens/ml-100k/u.item",header=None,sep='|',encoding='iso-8859-1')
print(movies.head(2))
movies.columns=["ID", "Title", "ReleaseDate", "VideoReleaseDate", "IMDB", "Unknown", "Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "FilmNoir", "Horror", "Musical", "Mystery", "Romance", "SciFi", "Thriller", "War", "Western"]
print(movies.head(2))
#Delete the columns which are not required
movies = movies[movies.columns.delete([0,2,3,4])]
print('Updated Data set is: ',movies.head(2))
#Delete the duplicate values
movies.drop_duplicates(inplace=True)
print('Shape of the data set is: ',movies.shape)

#Find the count of movies with Romance as genre
isRomance = (movies['Romance']==1)
print(movies[isRomance]['Title'].count())

# Simpler way is as below
print(movies.Romance.value_counts())

#Now count the movies with both Action and Drama as genres
is_action = movies['Action'] ==1
is_drama = movies['Drama']==1
print('Action and Drama Movie Count: ',movies[is_action & is_drama]['Title'].count())

#Other metod to perform above action
movies.groupby(['Action','Drama'])[['Drama','Action']].count()[['Drama']]

#Now calculate the pairwise distances
#Genre is not a dimension so use 19 columns while calculating the distances
dist = pdist(movies.iloc[:,1:20],'euclidean')
print('Pairwise distances are: ',dist)
print('Minimum distance is: ',dist.min())
print('Maximum distance is: ',dist.max())
print('Length of the array is: ',len(dist))
print('Standard deviation is: ',dist.std())

#Now  create the model for Hierarchical Clustering
model = hclust(affinity='euclidean',linkage='ward',n_clusters=10)
#Againt fit the 19 columns
model.fit(movies.iloc[:,1:20])
print(model.n_clusters)
#Check the labels attribute which is cluster label for each data point
print(model.labels_)

#Return the unique labels and count for it
print(np.unique(model.labels_,return_counts=True))

#Now create a Ward
Z = ward(movies.iloc[:,1:20])
print(Z)
#Draw a Dendogram
plt.figure(figsize=(8,8))
X = dendrogram(Z)


movies['clusters'] = model.labels_
print(movies.head(2))
#plt.show()
print(movies.groupby(by='clusters').mean())

#Now which in which cluster does movie Men in black belong to?
print(movies[movies['Title'].str.match('men in black',case = False)])
print(movies[movies['Title'] == 'Men in Black(1997)']['clusters'])

#Display 10 other movies that belong to same cluster as that of movie Men in Black
print(movies.Title[movies.clusters == 0].head(10))

#Now implement K-Means Clustering
km = kmeans(n_clusters=10,random_state=42)
km.fit(movies.iloc[:,1:20])

#Number of iterations it took algorithm to converge
print(km.n_iter_)
print(km.labels_)

#Save above labels in once column
movies['kmeansclust'] = km.labels_
print(movies.head(5))

#Sum of distances of samples to their closest cluster center
#Winthin Sum of Squares
print(km.inertia_)

#Number of movies in each cluster
print(pd.Series(km.labels_).value_counts().sort_index())
print(km.cluster_centers_)
#Cluster centers can be found out using group by mean as well
#Only positive values will be same as that of cluster centers
print(movies.groupby('kmeansclust').mean())

#Scree plot or Elbow plot
withinss = []
for i in range(2,20):
    km = kmeans(n_clusters=i,n_init=10,random_state=42)
    withinss.append(km.fit(movies.iloc[:,1:20]).inertia_)
plt.plot(range(2,20),withinss,'-o')
plt.xlabel('Number of clusters')
plt.ylabel('Total within SS')
plt.xlim(1,20)
plt.show()