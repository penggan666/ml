from __future__ import print_function
from scipy import io as spio
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data=spio.loadmat("data_digits.mat")
X=data['X']
y=data['y']
y=np.ravel(y)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # test_size为浮点数 则意思为随机抽取20%的数据作为测试集
# 特征归一化
scaler = StandardScaler()
scaler.fit_transform(x_train)
scaler.fit_transform(x_test)

# 逻辑回归
model = LogisticRegression()
model.fit(x_train, y_train)

predict = model.predict(x_test)  # 预测

print(u"预测准确度为：%f%%" % np.mean(np.float64(predict == y_test) * 100))