import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.metrics import mean_squared_error as mse

import matplotlib.pyplot as plt
import os
import sys

print(os.getcwd())
url=os.path.join(os.getcwd(),"..\\")
print(url)
os.chdir(url)

#Load data
all_train = pd.read_csv('data/preprocessed_dummy_train.csv')
all_valid = pd.read_csv('data/preprocessed_dummy_valid.csv')
alldata=all_train.append(all_valid)
alldata.head()

#Split Data set 

x_train = all_train.drop(columns = ['Id','SalePrice'])
y_train = all_train['SalePrice']
x_test = all_valid.drop(columns = ['Id','SalePrice'])
y_test = all_valid['SalePrice']

# x = alldata.drop(columns = ['Id','SalePrice'])
# y = alldata['SalePrice']
# x_train,x_test,y_train,y_test=train_test_split(x, y, train_size=0.8, random_state=4)

#Find the optimal k by calculating minimum rmse
results = []
k_list = np.arange(1,60)
for i in k_list:
    reg = KNN(n_neighbors = i)
    reg.fit(x_train, y_train)
    predict_test = reg.predict(x_test)
    
    results.append(mse(np.log(y_test), np.log(predict_test),squared = False))

index = np.argmin(results)   
print(index)
print(f"Best k: {k_list[index]}")
print(f"Best RMSE: {results[index]}")
print("Minimum error:",min(results),"at K =",results.index(min(results))+1)

#Elbow plot
plt.plot(k_list,results,color='blue', linestyle='dashed', 
         marker='o',markerfacecolor='red', markersize=5)
plt.xlabel("K Nearest Neighbors")
plt.ylabel("RMSE")
plt.title("Elbow Plot")
plt.show()


#predict test values using K = 1
all_test = pd.read_csv('data/normed_dummy_test.csv')
x_ttest = all_test.drop(columns = ['Id','SalePrice'])
y_ttest = all_test['SalePrice']

reg = KNN(n_neighbors = k_list[index],weights='distance')
reg.fit(x_train, y_train)
predict_y = reg.predict(x_ttest)
    
df = pd.DataFrame({'Id': all_test.Id,
                'SalePrice': predict_y})

df.to_csv('KNN/submission1.csv', index=False, header=True)