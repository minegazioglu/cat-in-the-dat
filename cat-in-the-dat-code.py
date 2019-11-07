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
logistic_regression_cv_scores = []
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
    
    # Remove outliers
    
    # X_train
    
    Q1 = X_train.quantile(0.02)
    Q3 = X_train.quantile(0.98)
    IQR = Q3 - Q1
    idx_train = ~((X_train < (Q1 - 1.5 * IQR)) |(X_train > (Q3 + 1.5 *  IQR))).any(axis=1)
    X_train = X_train[idx_train]
    # y_train
    y_train = y_train[idx_train]
    
    
    # X_test
    
    Q1 = X_test.quantile(0.02)
    Q3 = X_test.quantile(0.98)
    IQR = Q3 - Q1
    idx_test = ~((X_test < (Q1 - 1.5 * IQR)) |(X_test > (Q3 + 1.5 *  IQR))).any(axis=1)
    X_test = X_test[idx_test]
    # y_test
    y_test = y_test[idx_test]
    
    
    # Standard Scaler 
    sss = StandardScaler()
    ss.fit(X_train)
    X_train = ss.transform(X_train)
    X_test = ss.transform(X_test)
    # Algorithm Trials
    # Algorithm are commented out to be used when needed for convenience
    
    #1.LogisticRegressionCV
    
    lrcv = LogisticRegressionCV(cv=10,Cs = [0.01,0.1,1,10], 
                                random_state = 100, 
                                max_iter = 100,
                                penalty = "l2", 
                                solver = "liblinear", 
                                dual = False)
    lrcv.fit(X_train,y_train)
    predictions = lrcv.predict_proba(X_test)[:,1]
    
    #Roc auc score Evaluation
    score = roc_auc_score(y_test,predictions)
    roc_auc_scores.append(score)
    
    
    #2.XGBoost
    
    #xgb_al = XGBClassifier(n_estimators = 200, 
    #                       random_state = 42, 
    #                       scale_pos_weight = 2, 
    #                       booster = "gblinear",
    #                       learning_rate = 0.3,
    #                       min_child_weight = 1,
    #                       max_depth = 2,
    #                       gamma = 0,
    #                       subsample = 0.6,
    #                       colsample_bytree = 0.6,
    #                       reg_lambda = 1,
    #                       reg_alpha = 1e-5,
    #                       seed = 0,
    #                       cv = 10,
    #                       early_stopping_rounds=50,
    #                       scoring='roc_auc')
    #xgb_al.fit(X_train,y_train)
    #predictions = xgb_al.predict_proba(X_test)[:,1]
    
    #Roc auc score Evaluation
    #score = roc_auc_score(y_test,predictions)
    #roc_auc_scores.append(score)
    #logistic_regression_cv_scores.append(lrcv.scores_)
    
    #3.RandomForest
    #rf = RandomForestClassifier(bootstrap=True, 
    #                            class_weight=None, 
    #                            criterion='gini',
    #                            max_depth=9, 
    #                            max_features='auto', 
    #                            max_leaf_nodes=None,
    #                            min_impurity_decrease=0.0, 
    #                            min_impurity_split=None,
    #                            min_samples_leaf=1, 
    #                            min_samples_split=2,
    #                            min_weight_fraction_leaf=0.0, 
    #                            n_estimators=500, 
    #                            n_jobs=1,
    #                            oob_score=False, 
    #                            random_state=42, 
    #                            verbose=0, 
    #                            warm_start=False)
    #rf.fit(X_train,y_train)
    #predictions = rf.predict_proba(X_test)[:,1]
    #score = roc_auc_score(y_test,predictions)
    #roc_auc_scores.append(score)
    
    # 4.LightGBM
    
    #lgb_train = lgb.Dataset(X_train,y_train)
    #lgb_eval = lgb.Dataset(X_test,y_test,reference = lgb_train)
    #params = {  
    #    'boosting_type': 'gbdt',  
    #    'objective': 'binary',  
    #    'metric': {'binary_logloss', 'auc'},  
    #    'num_leaves':150, #former 150 0.7556
    #    'max_depth': 20, #20 > #25 
    #    'min_data_in_leaf': 200,#400
    #    'learning_rate': 0.01,  
    #    'feature_fraction': 0.95,  
    #    'bagging_fraction': 0.95,  
    #    'bagging_freq': 15, #20  
    #    'lambda_l1': 0,    
    #    'lambda_l2': 0, 
    #    'min_gain_to_split': 0.1,#former 0.1:0.7120680551013195
    #    'verbose': 0,  
    #    'is_unbalance': True  }  
    #
    #gbm = lgb.train(params,  
    #            lgb_train,  
    #            num_boost_round=10000,  
    #            valid_sets=lgb_eval,  
    #            early_stopping_rounds=700)  
    #
    #predictions= gbm.predict(X_test, num_iteration=gbm.best_iteration)
    #score = roc_auc_score(y_test,LGBM_TEST)
    #roc_auc_scores.append(score)
    
    
    # run Gridsearch on 4 different algorithm to tune their parameters
    
    # Gridsearch
    
    #param_test1 = {'parameter':[8,9,10]}
    
    #gr = GridSearchCV(algorithm,param_test1,verbose = 1,scoring='roc_auc')
    #gr.fit(X_train,y_train)
    #print(gr.cv_results_, gr.best_params_, gr.best_score_)

# take the average of the scores in roc_auc_list that will be the final_score
final_score = sum(roc_auc_scores) / len(roc_auc_scores)
print(final_score)


#final_scores for each algorithm:
# LogisticRegressionCV : 0.7867598254387023(13 mins)
# XGBoost              : 0.7832078833509699(13 min 33 seconds) 
# RandomForest         : 0.7428414695494446(1h 3min 49s)
