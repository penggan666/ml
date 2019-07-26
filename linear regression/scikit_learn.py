
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import numpy as np

data=np.loadtxt("ex1data1.txt",delimiter=",",dtype=np.float64)
X = data[:, 0:-1]  # X表示0到倒数第二列
y=data[:,-1]
#归一化操作
scaler=StandardScaler()
scaler.fit(X)
x_train=scaler.transform(X)
x_test = scaler.transform(np.array([6.1101]).reshape(1,-1))
#线性模型拟合
model=linear_model.LinearRegression()
model.fit(x_train,y)

#预测
result=model.predict(x_test)
print(result)
