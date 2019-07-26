from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize #scipy中的优化包
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)

def LogisticTRegression():
    data=np.loadtxt("ex2data2.txt",delimiter=",",dtype=np.float64)
    X=data[:,0:-1]
    y=data[:,-1]
    m=len(y)
    plot_data(X,y)#作图
    X = np.hstack((np.ones((m, 1)), X))  # 在X前加一列1
    initial_theta=np.zeros((X.shape[1],1))#shape1代表列数，shape0代表行数
    #初始化lamda，一般取0.01 0.1 1
    lamda = 0.1
    J=costFunction(initial_theta,X,y,lamda)#计算一下给定初始化的theta和lamda求出的代价J
    print(J)
    result=optimize.fmin_bfgs(costFunction,initial_theta,fprime=gradient,args=(X,y,lamda))
    #调用scipy中的优化算法fmin_bgs（拟牛顿法）
    #costFuncion是自己实现的一个求代价的函数
    #initial_theta表示初始化的值
    #fprime指定costFunction的梯度
    #args是其余参数，以元组的形式传入，最后会将最小化costFunction的theta返回
    p=predict(X,result)
    print(u'在训练集上的准确度为%f%%' % np.mean(np.float64(p == y) * 100))



#代价函数
def costFunction(initial_theta,X,y,inital_lamda):
    m=len(y)
    J=0

    h=sigmoid(np.dot(X,initial_theta)) #计算h函数
    theta1=initial_theta.copy() #因为正则化从1开始，不包含0，所以复制一份，设置theta【0】=0
    theta1[0]=0

    temp=np.dot(np.transpose(theta1),theta1)
    J=(-np.dot(np.transpose(y),np.log(h))-np.dot(np.transpose(1-y),np.log(1-h))+temp*inital_lamda/2)/m
    return J


#1/1+e^(-z)
def sigmoid(z):
    h=np.zeros((len(z),1))  #初始化，与z的长度一致
    h=1.0/(1.0+np.exp(-z))
    return h

#计算梯度
def gradient(initial_theta,X,y,initial_lamda):
    m=len(y)
    grad=np.zeros((initial_theta.shape[0]))
    h=sigmoid(np.dot(X,initial_theta))
    print(h.shape)
    theta1=initial_theta.copy()
    theta1[0]=0
    grad=np.dot(np.transpose(X),h-y)/m+initial_lamda/m*theta1
    return grad

#显示二维图形
def plot_data(X,y):
    pos=np.where(y==1) #找到y=1的坐标位置
    neg=np.where(y==0) #找到y=0的坐标位置
    #作图
    #plt.figure(figsize=(15,12))
    plt.plot(X[pos,0],X[pos,1],'ro')
    plt.plot(X[neg,0],X[neg,1],'bo')
    plt.title(u"两个类别的散点图",fontproperties=font)
    plt.show()

#预测
def predict(X,theta):
    m=X.shape[0]
    p=np.zeros((m,1))
    p=sigmoid(np.dot(X,theta))  #预测的结果是个概率值
    for i in range(m):
        if p[i]>0.5:
            p[i]=1
        else:
            p[i]=0
    return p

if __name__=="__main__":
    LogisticTRegression()

