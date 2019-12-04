import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")

train = pd.read_csv("Kaggle/catinthedat/train.csv")
test = pd.read_csv("Kaggle/catinthedat/test.csv")
sample_sub_df = pd.read_csv("Kaggle/catinthedat/sample_submission.csv")

train_id = train["id"]
test_id = test["id"]
target = train["target"]
train.drop(["id","target","bin_0"], axis = 1,inplace = True)
test.drop(["id","bin_0"], axis = 1,inplace = True)

# nom_0
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

bin34_dict = {"T":1,"F":0,"Y":1,"N":0}

#bin_3
train["bin_3"] = train["bin_3"].map(bin34_dict)
test["bin_3"] = test["bin_3"].map(bin34_dict)

#bin_4
train["bin_4"] = train["bin_4"].map(bin34_dict)
test["bin_4"] = test["bin_4"].map(bin34_dict)

# ord_5 Method Trials:
## Method 1

train["ord_5_1"] = train["ord_5"].str[0]
train["ord_5_2"] = train["ord_5"].str[1]

test["ord_5_1"] = test["ord_5"].str[0]
test["ord_5_2"] = test["ord_5"].str[1]


train.drop(["ord_5","ord_5_2"],axis = 1, inplace = True)
test.drop(["ord_5","ord_5_2"],axis = 1, inplace = True)

# Ascii Application

import string

for column in ["ord_3","ord_4","ord_5_1"]:
    train[column] = train[column].apply(lambda x: string.ascii_letters.index(x))
    test[column] = test[column].apply(lambda x: string.ascii_letters.index(x))

## Method 2

#ord_5 = sorted(list(set(train["ord_5"].values)))
#ord_5 = dict(zip(ord_5,range(len(ord_5))))

#train["ord_5"] = train["ord_5"].apply(lambda x: ord_5[x]).astype(int)
#test["ord_5"] = test["ord_5"].apply(lambda x: ord_5[x]).astype(int)

#ord_3 = sorted(list(set(train["ord_3"].values)))
#ord_3 = dict(zip(ord_3,range(len(ord_3))))

#train["ord_3"] = train["ord_3"].apply(lambda x: ord_3[x]).astype(int)
#test["ord_3"] = test["ord_3"].apply(lambda x: ord_3[x]).astype(int)

#ord_4 = sorted(list(set(train["ord_4"].values)))
#ord_4 = dict(zip(ord_4,range(len(ord_4))))

#train["ord_4"] = train["ord_4"].apply(lambda x: ord_4[x]).astype(int)
#test["ord_4"] = test["ord_4"].apply(lambda x: ord_4[x]).astype(int)

## Method 3

#from sklearn.preprocessing import OrdinalEncoder # Encode categorical features as an integer array

#ordinal_columns = ["ord_5","ord_3","ord_4"]

#orden = OrdinalEncoder(categories = "auto")
#orden.fit(train[ordinal_columns])
#train[ordinal_columns] = orden.transform(train[ordinal_columns])
#test[ordinal_columns]  = orden.transform(test[ordinal_columns])

# Replace values that are both not in train and test

columns_to_check = ["nom_7","nom_8","nom_9"]
#
for column in columns_to_check:
    values_to_replace = set(train[column]) ^ set(test[column])
    if values_to_replace:
        print("Column " + column + " has " + str(len(values_to_replace)) + " values")
        train[column] = train[column].apply(lambda x: "21578c358" if x in values_to_replace else x)
        test[column]  = test[column].apply(lambda x: "21578c358" if x in values_to_replace else x)
    else:
        print("Column" + column + "has no none existent values")
        
        
# ord_1

ord1_dict = {"Novice":1,"Contributor":2,"Expert":3,"Master":4,"Grandmaster":5}
train["ord_1"] = train["ord_1"].map(ord1_dict)

ord1_dict = {"Novice":1,"Contributor":2,"Expert":3,"Master":4,"Grandmaster":5}
test["ord_1"] = test["ord_1"].map(ord1_dict)

#ord_2

ord2_dict = {"Freezing":1,"Cold":2,"Warm":3,"Hot":4,"Boiling Hot":5,"Lava Hot":6}
train["ord_2"] = train["ord_2"].map(ord2_dict)

ord2_dict = {"Freezing":1,"Cold":2,"Warm":3,"Hot":4,"Boiling Hot":5,"Lava Hot":6}
test["ord_2"] = test["ord_2"].map(ord2_dict)

#Month & Day

# Month sin-cosin transformation

train['mnth_sin'] = np.sin((train.month-2)*(2.*np.pi/12.0))
train['mnth_cos'] = np.cos((train.month-2)*(2.*np.pi/12.0))


test['mnth_sin'] = np.sin((test.month-2)*(2.*np.pi/12.0))
test['mnth_cos'] = np.cos((test.month-2)*(2.*np.pi/12.0))

# Day sin-cosin transformation

train['day_sin'] = np.sin((train.day-2)*(2.*np.pi/7.0))
train['day_cos'] = np.cos((train.day-2)*(2.*np.pi/7.0))


test['day_sin'] = np.sin((test.day-2)*(2.*np.pi/7.0))
test['day_cos'] = np.cos((test.day-2)*(2.*np.pi/7.0))

train.drop(["month","day"], axis = 1,inplace = True)
test.drop(["month","day"], axis = 1,inplace = True)


# Convert nom_5 - nom_9 into integers

hexadecimal_columns = ["nom_5","nom_6","nom_7","nom_8","nom_9"]
import binascii

for column in hexadecimal_columns:
    train[column] = train[column].apply(lambda x: int(x,36))
    test[column] = test[column].apply(lambda x: int(x,36))

    
#dummy encoding
traintest = pd.concat([train,test])
dummies = pd.get_dummies(traintest,columns = traintest.columns, drop_first = True,sparse = True)

train_ohe = dummies.iloc[:train.shape[0],:]
test_ohe = dummies.iloc[train.shape[0]:,:]

train_ohe = train_ohe.sparse.to_coo().tocsr()
test_ohe = test_ohe.sparse.to_coo().tocsr()

# Submission


%%time
lr = LogisticRegression(C=0.123456789,
                        solver="lbfgs",
                        max_iter=70000,
                        random_state = 42,
                        intercept_scaling = 0.1,
                        tol=0.0002,
                        class_weight = "balanced")

lr.fit(train_ohe,target)


sample_sub_df['target'] =  lr.predict_proba(test_ohe)[:,1]
sample_sub_df.to_csv('subm85.csv', index=False)
        


# train score

folds = StratifiedKFold(n_splits = 5,shuffle = True,random_state = 42)

model = LogisticRegression(C=0.123456789,#C=0.12356789
                        solver="lbfgs",
                        max_iter=70000,
                        random_state = 42,
                        intercept_scaling =0.1,
                        tol=0.0002)

final_scores = []

for fold, (train_index,validation_index) in enumerate(folds.split(train_ohe,target,groups = target)):
    print("Fold:", fold+1)
    train_x,train_y = train_ohe[train_index,:], target[train_index]
    valid_x,valid_y = train_ohe[validation_index,:], target[validation_index]
    
    %time model.fit(train_x,train_y)
    
    predictions = model.predict(valid_x)
    
    final_scores.append(roc_auc_score(valid_y,predictions))
    print("AUC SCORE: {}".format(roc_auc_score(valid_y,predictions)))
    
    



