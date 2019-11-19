# import necessary packages

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
#from ml import simple #the1owl
import gc
import warnings
warnings.filterwarnings('ignore')

#importing train & test data

train = pd.read_csv("Kaggle/catinthedat/train.csv")
test = pd.read_csv("Kaggle/catinthedat/test.csv")
sample_sub_df = pd.read_csv("Kaggle/catinthedat/sample_submission.csv")

# get target column
target = train['target']
# train_id
train_id = train['id'],
# test_id
test_id = test['id']
# Drop target, id, bin_ 0 columns from train
train.drop(['target', 'id', 'bin_0'], axis=1, inplace=True)
# Drop target, id, bin_ 0 columns from test
test.drop(['id','bin_0'], axis=1, inplace=True)

#dictionary for encoding bin_3 & bin_4

bin34_dict = {"T":1,"F":0,"Y":1,"N":0}

# Encoding bin_3 and bin_4

train["bin_3"] = train["bin_3"].map(bin34_dict)
train["bin_4"] = train["bin_4"].map(bin34_dict)


test["bin_3"] = test["bin_3"].map(bin34_dict)
test["bin_4"] = test["bin_4"].map(bin34_dict)

# Ord_5 column is made up of dual combinations of letters without upper/lower case letter sensitivity(eg. "ab")
# Split dual letters into 2 columns 

# train
train["ord_5_1"] = train["ord_5"].str[0]
train["ord_5_2"] = train["ord_5"].str[1]


#test
test["ord_5_1"] = test["ord_5"].str[0]
test["ord_5_2"] = test["ord_5"].str[1]

# drop ord_5 column from train data
test.drop("ord_5",axis = 1,inplace= True)
train.drop("ord_5",axis = 1,inplace= True)
#train.drop("ord_5_2",axis = 1,inplace= True)
test.drop("ord_5_2",axis = 1,inplace= True)
train.drop("ord_5_2",axis = 1,inplace= True)

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


%%time

# Concat train and test data
traintest = pd.concat([train, test])
# Perform dummy encoding on traintest data
dummies = pd.get_dummies(traintest, columns=traintest.columns, drop_first=True, sparse=True)
# Split dummies to train and test based on indexes
train_ohe = dummies.iloc[:train.shape[0], :]
test_ohe = dummies.iloc[train.shape[0]:, :]

%%time
# convert sparse matrices to 
train_ohe = train_ohe.sparse.to_coo().tocsr()
test_ohe = test_ohe.sparse.to_coo().tocsr()

%%time
lr = LogisticRegression(C=0.123456789,#C=0.12356789
                        solver="liblinear",
                        max_iter=70000,
                        random_state = 42)

lr.fit(train_ohe,target)


sample_sub_df['target'] =  lr.predict_proba(test_ohe)[:,1]
sample_sub_df.to_csv('submission78.csv', index=False) # Score: 0.80789
