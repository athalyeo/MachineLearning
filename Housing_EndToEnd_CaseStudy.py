import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

housing = pd.read_csv("https://storage.googleapis.com/dimensionless/ML_with_Python/Housing.csv")

#Data preview
print(housing.head(2))
#Data Information
print(housing.info())
#Check Summary Statistics
print(housing.describe())
print('Shape of the data set: ',housing.shape)

#Data Visualization to gain insight
housing.hist(bins=50,figsize=(15,15))


#EDA Process
housing1 = housing.copy()
#Plot Geographic Information System plot
housing1.plot(kind = "scatter",x = "longitude",y="latitude",alpha = 0.1)


corr_matrix = housing1.corr()
print(corr_matrix)
# Scatter plot between median income and median house value
housing1.plot(kind = "scatter",x="median_income",y="median_house_value")

#Attribute Combinations / Feature Engineering - Deriving new features from existing features

#Calculate number of rooms per household
housing1["rooms_per_household"] = (housing1["total_rooms"] / housing1["households"])
#Ratio of bedrooms to total rooms
housing1["bedrooms_per_room"] = (housing1["total_bedrooms"] / housing1["total_rooms"])
#Ppulation per household
housing1["population_per_household"] = (housing1["population"] / housing1["households"])
print(housing1.shape)
print('Update Corelation Matrix',housing1.corr())

#Prepare the data for Machine Learning
housing_features = housing1.drop('median_house_value',axis = 1)
print(housing_features.columns)
print(housing_features.head(2))
housing_target = housing1['median_house_value']
print(housing_target.shape)

housing1 = housing1.drop('median_house_value',axis = 1)
print(housing1.columns)
#Check for missing values
print(housing1.isnull().sum())

#Replacing null values with desired value strategies
#If Distribution is normal then use Mean
# If Distribution is skewed then use Median because Medain is robust of central tendency
# For normal distribution Mean and Median are same so use Median
imputer = SimpleImputer(strategy="median")

#Creating data frame with only numerical features because imputer cannot be applied on Categorical features
housing_num = housing1.drop("ocean_proximity",axis = 1)
print(housing_num.columns)
#Fit the Imputer on Numerical Data
imputer.fit(housing_num)
#Median of the variables
print(imputer.statistics_)
#Carry out Imputation
transformed_values = imputer.transform(housing_num)
print(transformed_values)
housing_transformed = pd.DataFrame(transformed_values)
print(housing_transformed.head(2))

housing_transformed.columns = ['longitude','latitude','housing_median_age','total_rooms',
                               'total_bedrooms','population','households','median_income',
                               'rooms_per_household','bedrooms_per_room','population_per_household']
print(housing_transformed.head(2))
print(housing_transformed.isnull().sum())

#Handling Text and Categorical values
#One hot coding does the same thing as that of dummyifying process
cat_encoder = OneHotEncoder()
#Check the cardinality that is unique number of labels in categorical variables
print(len(housing['ocean_proximity'].unique()))
#Separating Categorical Variables
housing_cat = housing[['ocean_proximity']]
dummy_values = cat_encoder.fit_transform(housing_cat)
print(dummy_values)
print(dummy_values.toarray())
print(cat_encoder.categories_)
#Convert to Dataframe
housing_cat = pd.DataFrame(dummy_values.toarray())
print(housing_cat.head(2))

#Fetching column names from Categories attribute
housing_cat.columns = ['<1H OCEAN','INLAND','ISLAND','NEAR BAY','NEAR OCEAN']
print(housing_cat.head(2))

#Feature Scaling
std_scaler = StandardScaler()
std_scaler.fit(housing_num)
print(housing_num.describe())
scaled_values = std_scaler.transform(housing_num)
housing_scaled = pd.DataFrame(scaled_values)
housing_scaled.columns = ['longitude','latitude','housing_median_age','total_rooms',
                          'total_bedrooms','population','households','median_income',
                          'rooms_per_household','bedrooms_per_room','population_per_household']
print(housing_scaled.describe())

# Transformations Pipeline
num_pipeline = Pipeline([('imputer',SimpleImputer(strategy = "median")),
                         ('std_scaler',StandardScaler()),])
housing_num_tr = num_pipeline.fit_transform(housing_num)
housing_pipeline = pd.DataFrame(housing_num_tr)
housing_pipeline.columns = ['longitude','latitude','housing_median_age','total_rooms',
                          'total_bedrooms','population','households','median_income',
                          'rooms_per_household','bedrooms_per_room','population_per_household']
print(housing_pipeline.head(2))
print(housing_pipeline.describe())

# ColumnTransformer - To apply all transformation together on housing data set
num_attributes = list(housing_num)
print(num_attributes)
cat_attributes = ["ocean_proximity"]
print(cat_attributes)
print(housing_features.columns)
full_pipeline = ColumnTransformer([
    ("num",num_pipeline,num_attributes),
    ("cat",OneHotEncoder(),cat_attributes),
])
housing_prepared = full_pipeline.fit_transform(housing_features)
housing_prepared = pd.DataFrame(housing_prepared)
print(housing_prepared.shape)
print(housing_prepared.head(2))
housing_prepared.columns = ['longitude','latitude','housing_median_age','total_rooms',
                          'total_bedrooms','population','households','median_income',
                          'rooms_per_household','bedrooms_per_room','population_per_household',
                            '<1H OCEAN','INLAND','ISLAND','NEAR BAY','NEAR OCEAN']
print(housing_prepared.head(2))

#Create Test Set

X_train,X_test,y_train,y_test = train_test_split(housing_prepared,housing['median_house_value'],
                                                 test_size=0.2,random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#Select a  model and train it

lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
print('Score of the model is: ',lin_reg.score(X_train,y_train))
print('Beta values are: ',lin_reg.coef_)
y_pred_train = lin_reg.predict(X_train)
print(y_pred_train[:5])
lin_rmse = np.sqrt(mean_squared_error(y_train,y_pred_train))
print('RMSE value of the train model is: ',lin_rmse)

#Applying Decision Tree
tree_reg = DecisionTreeRegressor(max_depth=4)
tree_reg.fit(X_train,y_train)
print('Score of the Decision tree is: ',tree_reg.score(X_train,y_train))
tree_preds = tree_reg.predict(X_train)
tree_rmse = np.sqrt(mean_squared_error(y_train,tree_preds))
print('RMSE value of Decision tree model is: ',tree_rmse)

#Better Evaluation using Cross Validation
scores = cross_val_score(tree_reg,X_train,y_train,scoring="neg_mean_squared_error",cv=10)
tree_rmse_scores = np.sqrt(-scores)
print('Average of the tree rmse scores is: ',tree_rmse_scores.mean())

#apply CV on Linear model
scores = cross_val_score(lin_reg,X_train,y_train,scoring="neg_mean_squared_error",cv=10)
lin_rmse_scores = np.sqrt(-scores)
print("RMSE Scores for Linear model after apply CV : ",lin_rmse_scores)
print('Average of the Linear rmse scores is: ',lin_rmse_scores.mean())

#Apply Random Forest
forest_reg = RandomForestRegressor(random_state=100)
forest_reg.fit(X_train,y_train)
rforest_preds = forest_reg.predict(X_train)
rforest_rmse = np.sqrt(mean_squared_error(y_train,rforest_preds))
print('RMSE value of Random Forest Regressor model is: ',rforest_rmse)


#Fine Tune the model
param_grid = [
    {'n_estimators':[3,10,30],'max_features':[2,4,6,8]},
    {'bootstrap':[False],'n_estimators':[3,10],'max_features':[2,3,4]},
]

#Hyper Parameter Tuning
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg,param_grid,cv=5,scoring='neg_mean_squared_error',return_train_score=True)
grid_search.fit(X_train,y_train)
print('Best Parameter values are: ',grid_search.best_params_)
final_model = grid_search.best_estimator_

#Apply Final model on the Test data
final_predictions = final_model.predict(X_test)
final_rmse = np.sqrt(mean_squared_error(y_test,final_predictions))
print('Final value of rmse is: ',final_rmse)

#Representation
plt.scatter(y_test,final_predictions)
plt.xlabel("True")
plt.ylabel("Predicted")
plt.show()