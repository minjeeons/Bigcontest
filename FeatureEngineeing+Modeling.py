import pandas as pd
import numpy as np
import shap
from pycaret.classification import *
import math

Tree_merge = pd.read_csv('경로입력.csv')
train_x = Tree_merge[Tree_merge['is_applied'].notnull()]
test_x = Tree_merge[Tree_merge['is_applied'].isnull()]
test_x.drop(['is_applied'], axis = 1 , inplace = True)





print("===========================complete Read csv==========================")




train_x.loc[train_x['birth_year'] != train_x['birth_year'], 'birth_year'] = train_x['birth_year'].median()
#중앙값
train_x.loc[train_x['gender'] != train_x['gender'], 'gender'] = train_x['gender'].mode()[0]
#최빈값
train_x.loc[train_x['existing_loan_cnt'] != train_x['existing_loan_cnt'], 'existing_loan_cnt'] = train_x['existing_loan_cnt'].mean()
train_x.loc[train_x['credit_score'] != train_x['credit_score'], 'credit_score'] = train_x['credit_score'].mean()
train_x.loc[train_x['loan_limit'] != train_x['loan_limit'], 'loan_limit'] = train_x['loan_limit'].mean()
#평균값
temp = (train_x['loan_rate'].dropna().sample(train_x['loan_rate'].isnull().sum()))
temp.index = train_x[lambda x: x['loan_rate'].isnull()].index 
train_x.loc[train_x['loan_rate'].isnull(), 'loan_rate'] = temp


temp = (train_x['loan_limit'].dropna().sample(train_x['loan_limit'].isnull().sum()))
temp.index = train_x[lambda x: x['loan_limit'].isnull()].index

test_x.loc[test_x['birth_year'] != test_x['birth_year'], 'birth_year'] = test_x['birth_year'].median()
#중앙값
test_x.loc[test_x['gender'] != test_x['gender'], 'gender'] = test_x['gender'].mode()[0]
#최빈값

test_x.loc[test_x['existing_loan_cnt'] != test_x['existing_loan_cnt'], 'existing_loan_cnt'] = test_x['existing_loan_cnt'].mean()
test_x.loc[test_x['credit_score'] != test_x['credit_score'], 'credit_score'] = test_x['credit_score'].mean()
test_x.loc[test_x['loan_limit'] != test_x['loan_limit'], 'loan_limit'] = test_x['loan_limit'].mean()
#평균값
temp = (test_x['loan_rate'].dropna().sample(test_x['loan_rate'].isnull().sum()))
temp.index = test_x[lambda x: x['loan_rate'].isnull()].index # index 부여
test_x.loc[test_x['loan_rate'].isnull(), 'loan_rate'] = temp


temp = (test_x['loan_limit'].dropna().sample(test_x['loan_limit'].isnull().sum()))
temp.index = test_x[lambda x: x['loan_limit'].isnull()].index # index 부여








print("=====================complete fill NaN=======================")







train_x = train_x.drop(['application_id'], axis=1)
train_x = train_x.drop(['user_id'], axis=1)
train_x = train_x.drop(['product_id'], axis=1)

train_x['birth_year'] = 2022 - train_x['birth_year'] + 1 #나이로 변환, 나이대별로 인코딩해줄지는 나중에 생각

train_x['insert_year'] = train_x['insert_time'].apply(lambda x: x.split()[0].split('-')[0])
train_x['insert_month'] = train_x['insert_time'].apply(lambda x: x.split()[0].split('-')[1])
train_x['insert_day'] = train_x['insert_time'].apply(lambda x: x.split()[0].split('-')[2])


train_x[['insert_year', 'insert_month','insert_day']] = train_x[['insert_year', 'insert_month','insert_day']].apply(pd.to_numeric)
#생성일시 전처리

train_x['loanapply_insert_month'] = train_x['loanapply_insert_time'].apply(lambda x: x.split()[0].split('-')[1])
train_x['loanapply_insert_day'] = train_x['loanapply_insert_time'].apply(lambda x: x.split()[0].split('-')[2])
train_x.loc[:,['loanapply_insert_month']] = train_x.loc[:,['loanapply_insert_month']].astype('int32')
train_x.loc[:,['loanapply_insert_day']] = train_x.loc[:,['loanapply_insert_day']].astype('int32')
train_x[['loanapply_insert_month', 'loanapply_insert_day']] = train_x[['loanapply_insert_month', 'loanapply_insert_day']].apply(pd.to_numeric)




train_x['insert_-_loanapply_month'] = (train_x['insert_month'] - train_x['loanapply_insert_month'])
train_x['insert_-_loanapply_day'] = (train_x['insert_day'] - train_x['loanapply_insert_day'])
#생성 시기와 한도조회 시기가 모델에 영향을 줄것


train_x['credit_rate'] = train_x['credit_score'] / train_x['loan_rate']
train_x['credit_rate_log']=np.log(train_x['credit_score']/train_x['loan_rate'])
#신용점수가 높을수록 금리가 낮음

train_x['desired_rate'] = train_x['desired_amount'] / train_x['loan_rate']
#대출희망금액이 높은데 금리가 낮으면 잘 빌릴거임

train_x['limit_credit'] = train_x['loan_limit'] * train_x['credit_score']
train_x["limit_credit_log"] = train_x['limit_credit'].apply(lambda x : math.log(x+1))
train_x = train_x.drop(['limit_credit'], axis=1)
#한도와 신용점수 상관관계 높아서 곱해서 로그 취해주기

train_x['limit_rate'] = train_x['loan_limit'] * train_x['loan_rate']
train_x["limit_rate_log"] = train_x['limit_rate'].apply(lambda x : math.log(x+1))
train_x['limit_rate_log'][(train_x['limit_rate_log'] <= 0)] = 1
train_x = train_x.drop(['limit_rate'], axis=1)


train_x = train_x.drop(['insert_year'], axis=1)
train_x = train_x.drop(['company_enter_month'], axis=1)
train_x = train_x.drop(['insert_time'], axis=1)
train_x = train_x.drop(['loanapply_insert_time'], axis=1)




train_x['yearly_income'][(train_x['yearly_income'] <= 0)] = 1
train_x['desired_amount'][(train_x['desired_amount'] <= 0)] = 1
train_x['loan_limit'][(train_x['loan_limit'] <= 0)] = 1
train_x['credit_score'][(train_x['credit_score'] <= 0)] = 1

basic_col = train_x.columns
for col in basic_col:
    if col in ['yearly_income', 'desired_amount','loan_limit','credit_score']:
        train_x[col+"_log"] = train_x[col].apply(lambda x : math.log(x))







print("======================complete train_x Feature Engineering=======================")









test_x = test_x.drop(['application_id'], axis=1)
test_x = test_x.drop(['user_id'], axis=1)
test_x = test_x.drop(['product_id'], axis=1)

test_x['birth_year'] = 2022 - test_x['birth_year'] + 1 #나이로 변환, 나이대별로 인코딩해줄지는 나중에 생각


test_x['insert_year'] = test_x['insert_time'].apply(lambda y: y.split()[0].split('-')[0])
test_x['insert_month'] = test_x['insert_time'].apply(lambda y: y.split()[0].split('-')[1])
test_x['insert_day'] = test_x['insert_time'].apply(lambda y: y.split()[0].split('-')[2])



test_x[['insert_year', 'insert_month','insert_day']] = test_x[['insert_year', 'insert_month','insert_day']].apply(pd.to_numeric)
#생성일시 전처리

test_x['loanapply_insert_month'] = test_x['loanapply_insert_time'].apply(lambda y: y.split()[0].split('-')[1])
test_x['loanapply_insert_day'] = test_x['loanapply_insert_time'].apply(lambda y: y.split()[0].split('-')[2])
test_x.loc[:,['loanapply_insert_month']] = test_x.loc[:,['loanapply_insert_month']].astype('int32')
test_x.loc[:,['loanapply_insert_day']] = test_x.loc[:,['loanapply_insert_day']].astype('int32')
test_x[['loanapply_insert_month', 'loanapply_insert_day']] = test_x[['loanapply_insert_month', 'loanapply_insert_day']].apply(pd.to_numeric)


test_x['insert_-_loanapply_month'] = (test_x['loanapply_insert_month']-test_x['insert_month'] )
test_x['insert_-_loanapply_day'] = (test_x['loanapply_insert_day']-test_x['insert_day'] )
#생성 시기와 한도조회 시기가 모델에 영향을 줄것


test_x['credit_rate'] = test_x['credit_score'] / test_x['loan_rate']
test_x['credit_rate_log']=np.log(test_x['credit_score']/test_x['loan_rate'])
#신용점수가 높을수록 금리가 낮음

test_x['desired_rate'] = test_x['desired_amount'] / test_x['loan_rate']
#대출희망금액이 높은데 금리가 낮으면 잘 빌릴거임

test_x['limit_credit'] = test_x['loan_limit'] * test_x['credit_score']
test_x["limit_credit_log"] = test_x['limit_credit'].apply(lambda x : math.log(x+1))
test_x = test_x.drop(['limit_credit'], axis=1)
#한도와 신용점수 상관관계 높아서 곱해서 로그 취해주기

test_x['limit_rate'] = test_x['loan_limit'] * test_x['loan_rate']
test_x["limit_rate_log"] = test_x['limit_rate'].apply(lambda x : math.log(x+1))
test_x['limit_rate_log'][(test_x['limit_rate_log'] <= 0)] = 1
test_x = test_x.drop(['limit_rate'], axis=1)



test_x = test_x.drop(['insert_year'], axis=1)
test_x = test_x.drop(['company_enter_month'], axis=1)
test_x = test_x.drop(['insert_time'], axis=1)
test_x = test_x.drop(['loanapply_insert_time'], axis=1)

import math
test_x['yearly_income'][(test_x['yearly_income'] <= 0)] = 1
test_x['desired_amount'][(test_x['desired_amount'] <= 0)] = 1
test_x['loan_limit'][(test_x['loan_limit'] <= 0)] = 1
test_x['credit_score'][(test_x['credit_score'] <= 0)] = 1


basic_col = test_x.columns
for col in basic_col:
    if col in ['yearly_income', 'desired_amount','loan_limit','credit_score']:
        test_x[col+"_log"] = test_x[col].apply(lambda x : math.log(x))


print("======================complete test_x Feature Engineering=======================")


model = setup(train_x, target = 'is_applied', session_id = 42, silent = True, use_gpu = True, fold = 5, fold_strategy = 'stratifiedkfold',data_split_stratify = True)

model_dt = create_model('dt', fold = 5)

final_model = finalize_model(model_dt)

#final_model = load_model('Final_dt_model') ###Load Model

predict_y = predict_model(final_model, test_x)


test_x_a = Tree_merge[Tree_merge['is_applied'].isnull()]
test_x_a['is_applied'] = predict_y['Label']
test_x_a.drop(['birth_year', 'gender', 'insert_time', 'credit_score', 'yearly_income', 'income_type', 'company_enter_month', 'employment_type', 'houseown_type', 'desired_amount', 'purpose', 'existing_loan_cnt', 'loanapply_insert_time', 'bank_id', 'user_id', 'loan_limit', 'loan_rate', 'event'], axis = 1, inplace = True)
test_x_a.to_csv("./submit2.csv", index = False)
