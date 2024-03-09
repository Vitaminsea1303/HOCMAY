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
# max=0
# layer1=0
# layer2=0
# j=0
# for i in range(5,10):
#     for j in range(5,10):
#         clf = MLPRegressor(hidden_layer_sizes=(i,j),max_iter=1000,activation="relu",shuffle=False).fit(X_train,y_train)
#         y_predict = clf.predict(X_test) 
#         print('Coef of determination Neural Network: '+str(i)+' '+str(j), r2_score(y_test, y_predict))
#         if(r2_score(y_test, y_predict)>max):
#             max=r2_score(y_test, y_predict)
#             layer1=i
#             layer2=j
# print("do do cao nhat: "+str(layer1)+" "+str(layer2),max)


# Xác định mô hình cần tối ưu và các siêu tham số mà bạn muốn điều chỉnh.
# Xác định các giá trị có thể của mỗi siêu tham số và tạo thành một lưới (grid) các cấu hình tham số.
# GridSearchCV sẽ tạo ra một tập hợp các mô hình với từng cấu hình tham số trong lưới.
# Đối với mỗi mô hình, GridSearchCV sẽ đánh giá hiệu suất của mô hình đó bằng cách sử dụng 
# phương pháp cross-validation (thường là k-fold cross-validation) trên tập dữ liệu huấn luyện.
# Cuối cùng, GridSearchCV sẽ trả về mô hình có hiệu suất tốt nhất (thường là mô hình có độ chính xác 
# cao nhất hoặc độ đo tương tự tốt nhất) và các giá trị siêu tham số tương ứng.

# GridSearchCV là một công cụ trong thư viện scikit-learn của Python 
# dua vao Kfold va CrossValidation de tìm ra bộ siêu tham số tối ưu cho mô hình
# được sử dụng để tìm kiếm siêu tham số (hyperparameters) 
# tốt nhất cho một mô hình dự đoán.

from sklearn.model_selection import train_test_split,GridSearchCV
clf = MLPRegressor(max_iter=4000).fit(X_train,y_train)
params = {'hidden_layer_sizes': [(20,20),(60,60),(40,40,40),(24,16),(52,60)]}
mlp_cv_model = GridSearchCV(clf, params, cv = 5) # KFold - chia thành 5 phần 4 train - 1 test 
#clf la kieu mo hinh
#params Một từ điển (dictionary) chứa các siêu tham số
mlp_cv_model.fit(X_train, y_train)  #huan luyen theo mo hinh GridSearchCV
print(mlp_cv_model.best_params_)    #chon ra tham so tot nhat

# clf = MLPRegressor(hidden_layer_sizes=(60,60),max_iter=4000,activation="relu").fit(X_train,y_train)
# y_predict = clf.predict(X_test) 
# print('Coef of determination Neural Network: ', r2_score(y_test, y_predict))
# print('NSE Neural Network: ',NSE(y_test,y_predict))
# print('MAE Neural Network: ',mean_absolute_error(y_test, y_predict))
# print('RMSE Neural Network: ', np.sqrt(mean_squared_error(y_test, y_predict)),'\n')
