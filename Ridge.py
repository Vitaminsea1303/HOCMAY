import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
dataframe = pd.read_csv('Gold_Price.csv') 
dt_train,dt_test = train_test_split(dataframe,test_size=0.3,shuffle=False)

def NSE(y_test, y_pred):
    return (1 - (np.sum((y_pred - y_test) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)))
    
def MAE(y_test, y_pred):
    return mean_absolute_error(y_test, y_pred)

def RMSE(y_test, y_pred):
    return mean_squared_error(y_test, y_pred, squared=False)

X_train = dt_train.drop(['Date','Price'], axis = 1) 
y_train = dt_train['Price'] 
X_test= dt_test.drop(['Date','Price'], axis = 1)
y_test= dt_test['Price']

#Ridge 
clf = Ridge(alpha=1.0,max_iter=1000,tol=0.01)
rid = clf.fit(X_train,y_train)
y = np.array(y_test)    
y_pred = rid.predict(X_test)
# print("Thuc te Du doan Chenh lech")
# for i in range(0,len(y)):
#     print(" ",y[i]," ",y_pred[i]," ", abs(y[i]-y_pred[i]))
print("Coef of determination Ridge: ",r2_score(y_pred,y_test))
print("NSE Ridge: ", NSE(y_test,y_pred))
print('MAE Ridge:', MAE(y_test,y_pred))
print('RMSE Ridge:', RMSE(y_test,y_pred))