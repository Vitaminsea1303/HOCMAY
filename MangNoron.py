import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.linear_model import Perceptron 
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn import preprocessing 
from sklearn.neural_network import MLPRegressor
data = pd.read_csv('Gold_Price.csv') 
le=preprocessing.LabelEncoder() 
data=data.apply(le.fit_transform) 
dt_Train, dt_Test = train_test_split(data, test_size=0.3, shuffle = False) 
X_train = dt_Train.drop(['Date','Price'], axis = 1) 
y_train = dt_Train['Price'] 
X_test= dt_Test.drop(['Date','Price'], axis = 1)
y_test= dt_Test['Price']
y_test = np.array(y_test)
def NSE(y_test, y_pred):
    return (1 - (np.sum((y_pred - y_test) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)))
# clf = MLPClassifier(random_state=1, max_iter=2000, activation='logistic')
#vong for long nhau 
clf = MLPRegressor(hidden_layer_sizes=(10,10),max_iter=5000,activation="relu").fit(X_train,y_train)
#so lop an lop an thu nhat la 10 no tron, lop thu 2 co 10 no tron, ham kich hoat la relu, max_iter so lan lap toi da
#activation ham kich hoat lop an 
y_predict = clf.predict(X_test) 

# count = 0 
# for i in range(0,len(y_predict)) : 
#     if(y_test[i] == y_predict[i]) : 
#         count = count +1 
# print('Ty le du doan dung : ', count/len(y_predict))

# print('Accuracy: ',accuracy_score(y_test, y_predict))
print('Coef of determination Neural Network: ', r2_score(y_test, y_predict))
print('NSE Neural Network: ',NSE(y_test,y_predict))
print('MAE Neural Network: ',mean_absolute_error(y_test, y_predict))
print('RMSE Neural Network: ', np.sqrt(mean_squared_error(y_test, y_predict)),'\n')
