# import necessary packages
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

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

#nom_0_dict = {"Red":255,"Blue":255,"Green":128}
#train["nom_0"] = train["nom_0"].map(nom_0_dict)
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

#####

# Random Forest Region Indexes Encoding

# OHE of categorical columns

# Make a copy of feature dataframe X to use in OneHotEncoding and then later in RandomForest
OHE_X = X.copy()

#OHE on OHE_X
cat_columns = list(OHE_X.select_dtypes(include=['category','object']))
column_mask = []
for column_name in list(OHE_X.columns.values):
    column_mask.append(column_name in cat_columns)
le = LabelEncoder()
ohe = OneHotEncoder(categorical_features = column_mask)
for col in cat_columns:
    OHE_X[col] = le.fit_transform(OHE_X[col])
OHE_X = ohe.fit_transform(OHE_X)

# RandomForest
# Number of Trees: 5, Depth: 3 
rf = RandomForestClassifier(n_estimators=5, max_depth=3,random_state=0)
rf.fit(OHE_X, y)  
# Get region indexes
region_indexes = rf.apply(OHE_X)

# Add region indexes to X & drop all categorical columns for both train and test datasets
#train
X["regions"] = region_indexes.reshape(-1,1)
X.drop(list(X.select_dtypes(include=['category','object'])),axis = 1,inplace = True)
#test
new_test["regions"] = region_indexes.reshape(-1,1)
new_test.drop(list(new_test.select_dtypes(include=['category','object'])),axis = 1,inplace = True)

# LogisticRegressionCV

from sklearn.linear_model import LogisticRegressionCV
lrcv = LogisticRegressionCV(cv=5, random_state=0,max_iter = 500).fit(X, y)


sample_sub_df['target'] =  lrcv.predict_proba(new_test)[:,1]
sample_sub_df.to_csv('submission47.csv', index=False)

# Region indexes gives an array with shape (300000,5)



