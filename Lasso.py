import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

def NSE(y_test, y_pred):
    return (1 - (np.sum((y_pred - y_test) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)))
    
def MAE(y_test, y_pred):
    return mean_absolute_error(y_test, y_pred)

def RMSE(y_test, y_pred):
    return mean_squared_error(y_test, y_pred, squared=False)

dataframe = pd.read_csv('Gold_Price.csv') 
dt_train,dt_test = train_test_split(dataframe,test_size=0.3,shuffle=False)

X_train = dt_train.drop(['Date','Price'], axis = 1) 
y_train = dt_train['Price'] 
X_test= dt_test.drop(['Date','Price'], axis = 1)
y_test= dt_test['Price']
#Lasso
lasso = Lasso(alpha=1.0,max_iter=1000,tol=0.01).fit(X_train,y_train)
#
y_predict = lasso.predict(X_test)
y_test = np.array(y_test)

# print("Thuc te Du doan Chenh lech")
# for i in range(0,len(y_test)):
#     print(" ",y_test[i]," ",y_predict[i]," ", abs(y_test[i]-y_predict[i]))
print("Coef of determination Lasso :",r2_score(y_test,y_predict))
print("NSE Lasso: ", NSE(y_test,y_predict))
print('MAE Lasso:', MAE(y_test,y_predict))
print('RMSE Lasso:', RMSE(y_test,y_predict))