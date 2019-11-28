# import necessary packages
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")
import scipy.sparse as scipy
from scipy.sparse import coo_matrix, hstack, csr_matrix
from scipy import sparse
import gc
from sklearn.model_selection import GridSearchCV


# importing train & test data

train = pd.read_csv("Kaggle/catinthedat/train.csv")
test = pd.read_csv("Kaggle/catinthedat/test.csv")
sample_sub_df = pd.read_csv("Kaggle/catinthedat/sample_submission.csv")

##Encode bin_3 & bin_4
#dictionary for encoding bin_3 & bin_4
bin34_dict = {"T":1,"F":0,"Y":1,"N":0}

# Encoding bin_3 and bin_4
#train
train["bin_3"] = train["bin_3"].map(bin34_dict)
train["bin_4"] = train["bin_4"].map(bin34_dict)
#test
test["bin_3"] = test["bin_3"].map(bin34_dict)
test["bin_4"] = test["bin_4"].map(bin34_dict)

##Encoding nom_0

# Using RGB Color System -- map the numbers to the column then use OneHotEncoding [nom_0]
# Red -> (255,0,0)
# Blue -> (0,0,255)
# Green -> (0,128,0)
train = pd.get_dummies(train, columns = ["nom_0"])
test = pd.get_dummies(test, columns = ["nom_0"])

nom_0_Blue_dict = {0:0,1:255}
nom_0_Green_dict = {0:0,1:128}
nom_0_Red_dict = {0:0,1:255}
train["nom_0_Blue"] = train["nom_0_Blue"].map(nom_0_Blue_dict)
train["nom_0_Green"] = train["nom_0_Green"].map(nom_0_Green_dict)
train["nom_0_Red"] = train["nom_0_Red"].map(nom_0_Red_dict)

test["nom_0_Blue"] = test["nom_0_Blue"].map(nom_0_Blue_dict)
test["nom_0_Green"] = test["nom_0_Green"].map(nom_0_Green_dict)
test["nom_0_Red"] = test["nom_0_Red"].map(nom_0_Red_dict)

## Replace values that do not exist in both train and test

columns_to_test = ['nom_7', 'nom_8', 'nom_9']

replace_xor = lambda x: 'xor' if x in xor_values else x

for column in columns_to_test:
    xor_values = set(train[column].unique()) ^ set(test[column].unique())
    if xor_values:
        print('Column', column, 'has', len(xor_values), 'XOR values')
        train[column] = train[column].apply(replace_xor)
        test[column] = test[column].apply(replace_xor)
    else:
        print('Column', column, 'has no XOR values')

## Ord_1 & Ord_2
# Ord_1

ord1_dict = {"Novice":1,"Contributor":2,"Expert":3,"Master":4,"Grandmaster":5}
train["ord_1"] = train["ord_1"].map(ord1_dict)

ord1_dict = {"Novice":1,"Contributor":2,"Expert":3,"Master":4,"Grandmaster":5}
test["ord_1"] = test["ord_1"].map(ord1_dict)

# Ord_2

ord2_dict = {"Freezing":1,"Cold":2,"Warm":3,"Hot":4,"Boiling Hot":5,"Lava Hot":6}
train["ord_2"] = train["ord_2"].map(ord2_dict)

ord2_dict = {"Freezing":1,"Cold":2,"Warm":3,"Hot":4,"Boiling Hot":5,"Lava Hot":6}
test["ord_2"] = test["ord_2"].map(ord2_dict)

## Month & Day

# Month sin-cosin transformation

train['mnth_sin'] = np.sin((train.month-1)*(2.*np.pi/12.0))
train['mnth_cos'] = np.cos((train.month-1)*(2.*np.pi/12.0))


test['mnth_sin'] = np.sin((test.month-1)*(2.*np.pi/12.0))
test['mnth_cos'] = np.cos((test.month-1)*(2.*np.pi/12.0))

# Day sin-cosin transformation

train['day_sin'] = np.sin((train.day-1)*(2.*np.pi/6.0))
train['day_cos'] = np.cos((train.day-1)*(2.*np.pi/6.0))


test['day_sin'] = np.sin((test.day-1)*(2.*np.pi/6.0))
test['day_cos'] = np.cos((test.day-1)*(2.*np.pi/6.0))

# Drop stated column from train and test dataframes

new_train = train.drop(["id","target","day","month"], axis = 1)
new_test = test.drop(["id","day","month"], axis = 1)

# Split train into feature and target variables
X = new_train
y = train["target"]

####################

# Random Forest Region Indexes Encoding

# Only filter categorical columns for X and test datasets
categorical_columns_X = X.select_dtypes(include=['category','object'])
categorical_columns_new_test = new_test.select_dtypes(include=['category','object'])

# Filter numerical columns
numerical_columns_X = X[X.columns.difference(list(categorical_columns_X))]
numerical_columns_new_test = new_test[new_test.columns.difference(list(categorical_columns_new_test))]

le = LabelEncoder()
# OHE of categorical columns
# Make a copy of feature dataframe X to use in OneHotEncoding and then later in RandomForest
OHE_X = categorical_columns_X.copy()
OHE_X_test = categorical_columns_new_test.copy()

cat_columns = list(OHE_X.select_dtypes(include=['category','object']))
for col in cat_columns:
    le.fit(OHE_X[col])
    OHE_X[col] = le.transform(OHE_X[col])
    OHE_X_test[col] = le.transform(OHE_X_test[col])
    
ohe = OneHotEncoder()
ohe.fit(OHE_X)
OHE_X = ohe.transform(OHE_X)
OHE_X_test = ohe.transform(OHE_X_test)

# RandomTreesEmbedding
#from sklearn.ensemble import RandomTreesEmbedding
#rt = RandomTreesEmbedding(n_estimators = 100, max_depth = 14, random_state = 0)
#rt fit
#rt.fit(OHE_X,y)
# Region indeces for categorical_columns_X and categorical_columns_new_test
#region_indexes = rt.apply(OHE_X)
#region_indexes_test = rt.apply(OHE_X_test)


# SVD to reduce dimensions

#from sklearn.decomposition import TruncatedSVD
# Reduce down to 10 components
#svd = TruncatedSVD(n_components=10)
#svd.fit(train_encoding)
#train_encoding_reduced = svd.transform(train_encoding)
#test_encoding_reduced = svd.transform(test_encoding)

# RandomForest
# Number of Trees: 5, Depth: 3 
rf = RandomForestClassifier(n_estimators=100, max_depth=14,random_state=0)
rf.fit(OHE_X, y)  

# get the rf predictions (for future purposes)
test_rf_predictions=rf.predict_proba(OHE_X_test)[:,1]  
# Get region indexes
region_indexes = rf.apply(OHE_X)
region_indexes_test = rf.apply(OHE_X_test)
enc = OneHotEncoder(categorical_features='all',handle_unknown='ignore')
enc.fit(region_indexes)

train_encoding=enc.transform(region_indexes)
test_encoding=enc.transform(region_indexes_test)

# turn numerical columns to sparse matrices

numerical_X_sparse = scipy.csr_matrix(numerical_columns_X.values)
numerical_test_sparse = scipy.csr_matrix(numerical_columns_new_test.values)

# Concat sparse matrices

from scipy.sparse import hstack
processed_train = hstack((train_encoding, numerical_X_sparse))

processed_test = hstack((test_encoding, numerical_test_sparse))


###

# LogisticRegressionCV

from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
#lrcv = LogisticRegressionCV(cv=5, random_state=0,max_iter = 500,penalty='l1',solver = "liblinear",Cs = [0.123456789]).fit(processed_train, y) 
lr = LogisticRegression(C=0.1234, solver="lbfgs", max_iter=10000,random_state = 42,intercept_scaling = 0.1,verbose=1).fit(processed_train,y)


sample_sub_df['target'] =  lr.predict_proba(processed_test)[:,1]
sample_sub_df.to_csv('submission.csv', index=False)


# GridSearch
# param_test1 = {'n_estimators': [50,60,70,80,90,100]}
#gr = GridSearchCV(RandomForestClassifier(random_state=0, max_depth = 14),
#                        param_test1,
#                        verbose = 1,
#                        scoring='roc_auc')
#gr.fit(OHE_X,y)
#print(gr.cv_results_, gr.best_params_, gr.best_score_)
