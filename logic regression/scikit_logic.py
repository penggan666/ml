from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

def logisticRegression():
    data=np.loadtxt("ex2data1.txt",delimiter=",",dtype=np.float64)
    X=data[:,0:-1]
    y=data[:,-1]

    #train_test_split可将数据集随机划分为训练集和测试集
    x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2) #test_size为浮点数 则意思为随机抽取20%的数据作为测试集

    #特征归一化
    scaler=StandardScaler()
    scaler.fit_transform(x_train)
    scaler.fit_transform(x_test)

    #逻辑回归
    model=LogisticRegression()
    model.fit(x_train,y_train)

    #预测
    predict=model.predict(x_test)
    right=sum(predict==y_test)
    predict=np.hstack((predict.reshape(-1,1),y_test.reshape(-1,1)))
    print(predict)
    print('测试集准确率:%f%%'%(right*100.0/predict.shape[0]))

if __name__=='__main__':
    logisticRegression()