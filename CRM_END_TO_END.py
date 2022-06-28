import warnings
warnings.simplefilter(action='ignore',category=UserWarning)
warnings.simplefilter(action='ignore',category=FutureWarning)

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
pd.options.display.float_format = '{:.3f}'.format
from sklearn.preprocessing import LabelEncoder

# Importing Dataset
train = pd.read_csv("https://dim-mlpython.s3.amazonaws.com/CreditRiskModeling/train.csv", low_memory=False)
test = pd.read_csv("https://dim-mlpython.s3.amazonaws.com/CreditRiskModeling/test.csv", low_memory=False)
print(train.columns)
print(train.head(2))

# For each column heading we replace " " and convert the heading in lowercase
cleancolumn = []
for i in range(len(train.columns)):
    cleancolumn.append(train.columns[i].replace(' ', '_').lower())
train.columns = cleancolumn
print(train.head(2))
print("Display Summary  statistics")
print("Shape of the data set is ",train.shape)
print(train.describe())

#Removing the duplicates
unique_loanid = train['loan_id'].unique().tolist()
print('Total Samples of Data: ',str(train.shape[0]))
print('Total Unique samples in data: ',str(len(unique_loanid)))
print('Duplicate Samples in Data: ',str(train.shape[0] - len(unique_loanid)))

#Drop the duplicates
train = train.drop_duplicates()
print('Total Samples of Data: ',str(train.shape[0]))
print('Total Unique samples in data: ',str(len(unique_loanid)))
print('Duplicate Samples in Data: ',str(train.shape[0] - len(unique_loanid)))

#Get the Duplicates
dup_loanid = train[train.duplicated(['loan_id'],keep = False)]
print(dup_loanid.shape)
print(dup_loanid.describe())

#Sort the duplicated data frame in ascending order with NAs in the last
sorted_df = dup_loanid.sort_values(['current_loan_amount','credit_score'],ascending=True,na_position='last')
print(sorted_df.head(2))

#Consider the samples which are genuine
correct_df = sorted_df.drop_duplicates(['loan_id'],keep ='first')
print(correct_df.shape)
print(correct_df.head(2))

#Dropping the duplicate loan ids
train.drop_duplicates(['loan_id'],keep=False,inplace=True)
print(train.shape)

#Getting the final train data which is genuine
train = train.append(correct_df,ignore_index=True)
print('Shape of correct data set is ',train.shape)
print('Statistics of correct data set is: ',train.describe())
print(train["loan_status"].value_counts())

#Baseline accuracy of the model
baseline_accuracy = round(44616/(44616+17621)*100,2)
print('Baseline accuracy of the model is: ',baseline_accuracy)

#Preprocesing / Cleaning the data
print(train['years_in_current_job'].unique())
train['years_in_current_job'] = [0 if str(x) == '< 1 year' else x if str(x) == 'nan' else int(re.findall(r'\d+', str(x))[0]) for x in train['years_in_current_job']]
print(train['years_in_current_job'].unique())

#Clean Credit Score column - Range of credit scores is from 0-800, but some values are out of the range
def credit_range(x):
    if x > 800:
        return int(x/10)
    elif str(x) == 'nan':
        return x
    else:
        return int(x)

train['credit_score'] = train['credit_score'].map(credit_range)
print(train['credit_score'].head(2))

#Check for Maximum Open Credit
print(train.shape)
print(train[train['maximum_open_credit']=='#VALUE!'])
train =train[train['maximum_open_credit']!='#VALUE!']
train['maximum_open_credit'] = pd.to_numeric(train['maximum_open_credit'])
print(train.shape)

#Now clean up Monthly debt column - It has currency symbol and so datatype is String. Convert it to Numeric
train['monthly_debt']=train['monthly_debt'].str.strip('$') #Stripping off the $ sign
train['monthly_debt']=pd.to_numeric(train['monthly_debt']) #Converting it into a numerical variable
print(train['monthly_debt'].describe())

#Outlier treatment for Current Loan amount
ax = sns.boxplot(data=train['current_loan_amount'],orient="h",palette="Set2")

#check the description there is a placeholder in max value
print(train[train['current_loan_amount'] == 99999999.000])
#There are such 5861 samples, which is not low so need to replace it by NA's
train['current_loan_amount'] = [np.nan if int(x)==99999999 else x for x in train['current_loan_amount']]

ax = sns.boxplot(data=train['current_loan_amount'],orient="h",palette="Set2")

#Outlier treatment for Annual Income
ax = sns.boxplot(data=train['annual_income'],orient="h",palette="Set2")
print(train[train['annual_income']==8713547.000])
train = train[train['annual_income']!=8713547.000]
print(train.shape)
print(train[train['annual_income']>1200000])
train = train.drop([3686, 11660, 46615])
print(train.describe)

#Outlier treatment for Credit History
ax = sns.boxplot(data=train['years_of_credit_history'],orient="h",palette="Set2")
print(train[train['years_of_credit_history']>58])

#Outlier treatment for Number of Open Accounts
ax = sns.boxplot(data=train['number_of_open_accounts'],orient="h",palette="Set2")
print(train[train['number_of_open_accounts']>50])
ax = sns.boxplot(data=train['number_of_open_accounts'],orient="h",palette="Set2")

#Capping the outliers
IQR = train['number_of_open_accounts'].quantile(0.75) - train['number_of_open_accounts'].quantile(0.25)
upper_limit = train['number_of_open_accounts'].quantile(0.75) + (IQR * 1.5)
#lower_limit = train['number_of_open_accounts'].quantile(0.25) - (IQR * 1.5)
print("Upper Limit:", upper_limit)
train['number_of_open_accounts'] = [23.0 if ( x>23.0 and x!=np.nan) else x for x in train['number_of_open_accounts']]
ax = sns.boxplot(data=train['number_of_open_accounts'],orient="h",palette="Set2")

#Now check the missing  values
print(train.isnull().sum())

#Missing value treatment
#Bankruptcies and tax liens
train = train.dropna(subset = ['bankruptcies','tax_liens'])
print(train.shape)
print(train.isnull().sum())

#Months since last Delinquent
train['months_since_last_delinquent'].fillna(0,inplace=True)
print(train.shape)
print(train.isnull().sum())

#Applying 'Iterative Imputer' using default estimator 'Bayesian Ridge' which is Regularized Linear Regression.

train.reset_index(drop=True, inplace=True)
my_imputer = IterativeImputer() #Creating an instance of iterative imputer
#For this we need only numerical variables so filtering this
train_numerical = train._get_numeric_data() #extracting only the numerical columns
train_numerical_columns = train_numerical.columns #Saving the columns names for numeric data
print(train_numerical.shape) #dimension of the numeric dataset
print(train_numerical.isnull().sum()) #Number of missing values

train_imputed = my_imputer.fit_transform(train_numerical)
#Imputer will give the array as an object so need to convert it to Dataframe with columns
train_imputed = pd.DataFrame(train_imputed,columns=train_numerical_columns)
print(train_imputed.isnull().sum())

print(train_imputed.shape)
print(train.shape)

#Replace the values from Train set with Imputed values
train['years_in_current_job'] = train_imputed['years_in_current_job']
train['current_loan_amount'] = train_imputed['current_loan_amount']
train['credit_score'] = train_imputed['credit_score']
train['annual_income'] = train_imputed['annual_income']
print(train.shape)
print(train.isnull().sum())

#Converting months since last delinquent into categories
train['months_since_last_delinquent'] = ['extreme_risk' if x>51
        else 'high_risk' if x>32
        else 'moderate_risk' if x>16
        else 'low_risk' if x>0 else 'no_risk' for x in train['months_since_last_delinquent']]

print(train['months_since_last_delinquent'].unique())
#Drop Loan Id and Customer Id
train.drop(['loan_id', 'customer_id'], axis=1, inplace=True)

#Handling Categorical variables
le = LabelEncoder()
train['loan_status'] = le.fit_transform(train.loan_status)
print(le.classes_)

print(train.loan_status.unique())
print(train.term.unique())
print(train.home_ownership.unique())
train['home_ownership'] = ['Mortgage' if 'Mortgage' in x else x for x in train['home_ownership']]
print(train.home_ownership.unique())
print(train.purpose.unique())
train.drop(['purpose'], axis=1, inplace=True)

print(train.months_since_last_delinquent.unique())
cols_to_transform = ['term', 'months_since_last_delinquent', 'home_ownership']
train_with_dummies = pd.get_dummies(train, prefix=cols_to_transform)