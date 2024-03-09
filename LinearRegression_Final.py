import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


def NSE(y_test, y_predict):
    return (1 - (np.sum((y_predict - y_test) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)))

def MAE(y_test, y_predict):
    return mean_absolute_error(y_test, y_predict)

def RMSE(y_test, y_predict):
    return mean_squared_error(y_test, y_predict, squared=False)

dataframe = pd.read_csv('Gold_Price.csv') 
dt_train,dt_test = train_test_split(dataframe,test_size=0.3,shuffle=False)

# print("Train set:\n", dt_train)
# print("Test set:\n", dt_test)

X_train = dt_train.drop(['Date','Price'], axis = 1) 
y_train = dt_train['Price'] 
X_test= dt_test.drop(['Date','Price'], axis = 1)
y_test= dt_test['Price']

reg = LinearRegression(fit_intercept=False).fit(X_train,y_train)
#de ko di qua goc toa do de + hang so b
y_predict = reg.predict(X_test)
y_test = np.array(y_test)

print("Thuc te Du doan Chenh lech")
for i in range(0,len(y_test)):
    print(" ",y_test[i]," ",y_predict[i]," ", abs(y_test[i]-y_predict[i]))
print("Coef of determination LinearRegression :",r2_score(y_test,y_predict))
print("NSE LinearRegression: ", NSE(y_test,y_predict))
print('MAE LinearRegression:', MAE(y_test,y_predict))
print('RMSE LinearRegression:', RMSE(y_test,y_predict))