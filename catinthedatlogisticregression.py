#importing necessary packages
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import string
import math
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

#importing train & test data
train = pd.read_csv("Kaggle/catinthedat/train.csv")
test = pd.read_csv("Kaggle/catinthedat/test.csv")

# take a look at the data
#train
train.head()
#test
test.head()

# Categorical Encoding Columns

# Bin 3 & Bin 4
# dictionary for encoding bin_3 & bin_4
bin34_dict = {"T":1,"F":0,"Y":1,"N":0}
# Encoding bin_3 and bin_4
# bin_3
train["bin_3"] = train["bin_3"].map(bin34_dict)
train["bin_4"] = train["bin_4"].map(bin34_dict)
# bin_4
test["bin_3"] = test["bin_3"].map(bin34_dict)
test["bin_4"] = test["bin_4"].map(bin34_dict)

# Num_0

# Using RGB Color System -- map the numbers to the column then use OneHotEncoding [nom_0]
# Red -> (255,0,0)
# Blue -> (0,0,255)
# Green -> (0,128,0)

# use get_dummies to separate colors red,blue and green into 3 columns
train = pd.get_dummies(train, columns = ["nom_0"])
test = pd.get_dummies(test, columns = ["nom_0"])
# dictionaries that contain the values to be mapped for red,blue and green columns
nom_0_Blue_dict = {0:0,1:255}
nom_0_Green_dict = {0:0,1:128}
nom_0_Red_dict = {0:0,1:255}
# map dictionaries into train and test datasets
#train
train["nom_0_Blue"] = train["nom_0_Blue"].map(nom_0_Blue_dict)
train["nom_0_Green"] = train["nom_0_Green"].map(nom_0_Green_dict)
train["nom_0_Red"] = train["nom_0_Red"].map(nom_0_Red_dict)
#test
test["nom_0_Blue"] = test["nom_0_Blue"].map(nom_0_Blue_dict)
test["nom_0_Green"] = test["nom_0_Green"].map(nom_0_Green_dict)
test["nom_0_Red"] = test["nom_0_Red"].map(nom_0_Red_dict)

# Ord_1 & Ord_2

# Ord_1
# create a dict to later map wanted values to the dataset
#train
ord1_dict = {"Novice":1,"Contributor":2,"Expert":3,"Master":4,"Grandmaster":5}
train["ord_1"] = train["ord_1"].map(ord1_dict)
#test
ord1_dict = {"Novice":1,"Contributor":2,"Expert":3,"Master":4,"Grandmaster":5}
test["ord_1"] = test["ord_1"].map(ord1_dict)

# Ord_2
# create a dict to later map wanted values to the dataset
#train
ord2_dict = {"Freezing":1,"Cold":2,"Warm":3,"Hot":4,"Boiling Hot":5,"Lava Hot":6}
train["ord_2"] = train["ord_2"].map(ord2_dict)
#test
ord2_dict = {"Freezing":1,"Cold":2,"Warm":3,"Hot":4,"Boiling Hot":5,"Lava Hot":6}
test["ord_2"] = test["ord_2"].map(ord2_dict)

# Day & Month

# Month sin-cosin transformation
#train
train['mnth_sin'] = np.sin((train.month-1)*(2.*np.pi/12.0))
train['mnth_cos'] = np.cos((train.month-1)*(2.*np.pi/12.0))
#test
test['mnth_sin'] = np.sin((test.month-1)*(2.*np.pi/12.0))
test['mnth_cos'] = np.cos((test.month-1)*(2.*np.pi/12.0))

# Day sin-cosin transformation
#train
train['day_sin'] = np.sin((train.day-1)*(2.*np.pi/6.0))
train['day_cos'] = np.cos((train.day-1)*(2.*np.pi/6.0))
#test
test['day_sin'] = np.sin((test.day-1)*(2.*np.pi/6.0))
test['day_cos'] = np.cos((test.day-1)*(2.*np.pi/6.0))

# drop id,target,day,month columns from train data
new_train = train.drop(["id","target","day","month"], axis = 1)
# drop id column from train data
new_test = test.drop(["id"], axis = 1, inplace = True)

# Assign X,y variables to be used in StratifiedKFold to create train and test datasets from train data
X = new_train
y = train["target"]


# create StratifiedKFold object with 10 splits
skf = StratifiedKFold(n_splits=10)
# empty list to fill in roc_auc_scores for all 10 splits
roc_auc_scores = []
# splitting X,y
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    # add id column to use later for our calculations 
    X_test["id"] = X_test.index
    X_train["id"] = X_train.index
    # for creating a dictionary to map 1/total target values concatenate X_train,y_train
    basic_train = pd.concat([X_train,y_train], axis = 1)
    # columns we will apply target encoding to
    columns_p = ['nom_1', 'nom_2','nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9','ord_3', 'ord_4', 'ord_5']
    #for each column in columns_p groupby by column and target and count id
    for column in columns_p:
        basic_df = basic_train.groupby([column,"target"]).count()["id"].reset_index()
        #set column as index     
        df = basic_df.set_index(column)
        # count total of 1 labeled targets for each unique value in the column
        one_count = df.loc[df.target.eq(1), 'id']
        # count total of 0 labeled targets for each unique value in the column
        zero_count = df.loc[df.target.eq(0), 'id']
        # take the ratio one_count/(one_count+zero_count) if zero count is #NA replace with 0
        target_ratio = one_count.divide(one_count.add(zero_count, fill_value=0)).fillna(0)
        # create a dictionary from target_ratio dataframe
        df_dict = target_ratio.to_dict()
    
        #map dictionaries to X_train and X_test
        X_train[column] = X_train[column].map(df_dict)
        X_test[column] = X_test[column].map(df_dict)
    
    # drop id columns one more time
    X_train.drop(["id"], axis = 1, inplace = True)
    X_test.drop(["id"], axis = 1, inplace = True)
    # if any #NA values fill with mean of the column
    X_train.fillna(X_train.mean(),inplace = True)
    X_test.fillna(X_train.mean(),inplace = True)
    
    # Standard Scaler 
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.fit_transform(X_test)

    #LogisticRegressionCV
    
    lrcv = LogisticRegressionCV(cv=10,Cs = 1, random_state = 42)
    lrcv.fit(X_train,y_train)
    predictions = lrcv.predict_proba(X_test)[:,1]
    score = roc_auc_score(y_test,predictions)
    #append scores to roc_auc_scores list
    roc_auc_scores.append(score)
    #logistic_regression_cv_scores.append(lrcv.scores_)

# take the average of the scores in roc_auc_list that will be the final_score
final_score = sum(roc_auc_scores) / len(roc_auc_scores)
print(final_score)


