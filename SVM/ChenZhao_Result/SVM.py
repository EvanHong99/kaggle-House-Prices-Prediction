# -*- coding=utf-8 -*-
# @File     : SVM.py
# @Time     : 2022/7/23 14:45
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : house-prices-advanced-regression-techniques
# @Description:


#%%

from sklearn import svm
import pandas as pd
import numpy as np

XY_train = pd.read_csv('data/prepossess_train.csv',index_col=0,header=0)
X_train = XY_train.iloc[:,:-1]
y_train = XY_train.iloc[:,-1]
XY_test = pd.read_csv('data/prepossess_valid.csv',index_col=0,header=0)
X_test = XY_test.iloc[:,:-1]
y_test = XY_test.iloc[:,-1]

#%%

#clf = make_pipeline(Normalizer(), MinMaxScaler(), svm.SVC(kernel='linear'))
clf = svm.SVC(kernel='poly')
# clf = svm.SVR(kernel='poly')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred

#%%

result = pd.DataFrame({"The_log_of_True":(np.log(y_test)), "The_log_of_Pred": (np.log(y_pred))})
result

#%%

from sklearn.metrics import mean_squared_error
MSE = mean_squared_error(np.log(y_test), np.log(y_pred))
RMSE = np.sqrt(MSE)
print("RMSE:",RMSE)

#%%

result_data = pd.read_csv('data/prepossess_test.csv',index_col=0,header=0)
result_data = result_data.iloc[:,:-1]
result = clf.predict(result_data)
result = pd.DataFrame({'Id': result_data.index,'SalePrice':result})

#%%

result.set_index('Id',drop = True, inplace = True)

#%%

result.to_csv('mySubmission.csv')
