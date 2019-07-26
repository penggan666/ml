from __future__ import print_function
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import optimize #scipy中的优化包
import scipy.io as spio
from matplotlib import pyplot as plt#2D绘图库
from matplotlib.font_manager import FontProperties#解决字体问题
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)

def linearRegression():#学习率为0.01，迭代次数为400次
    print(u"加载数据。。。\n")#加个u表示对字符串进行unicode编码
    '''
    data=np.loadtxt("simple_ex1.txt",delimiter=",",dtype=np.float64)
    X=data[:,0:-1]#X表示0到倒数第二列
    y=data[:,-1]#y对应最后一列
    '''
    data=spio.loadmat("ex5data1.mat")
    X=data['X']
    y=data['y']

    m=len(y)
    y=np.ravel(y)#将y降为1维
    print(y)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    m1=len(y_train)#训练集的数据条数
    m2=len(y_test) #测试集的数据个数

    # 在X前加一列1
    x_train1=ployfeatures(x_train,2)
    x_train1,mu,sigma=featureNormaliza(x_train1)#归一化
    x_train1 = np.hstack((np.ones((m1, 1)), x_train1))
    x_train = np.hstack((np.ones((m1, 1)), x_train))
    x_test=np.hstack((np.ones((m2,1)),x_test))
    theta = np.zeros((x_train1.shape[1], 1))
    lamda=100
    final_theta = optimize.fmin_bfgs(computerCost, theta, fprime=gradientDescent, args=(x_train1, y_train, lamda))
    predict_x_train=predict(mu,sigma,final_theta,X)  #在训练集上进行预测
    print(predict_x_train)
    #predict_x_test=np.dot(x_test,final_theta)  #在测试集上进行预测
    '''画出训练集上的拟合情况'''
    plt.scatter(X,y)
    plt.plot(X,predict_x_train)
    plt.show()
    '''画出测试集上的拟合情况''''''
    plt.scatter(x_test[:,1],y_test)
    plt.plot(x_test[:,1],predict_x_)
    plt.show()
    '''

    #return mu,sigma,theta #返回均值mu，标准差sigma，和学习的结果theta

#增加特征项来解决欠拟合
def ployfeatures(X,p):
    x_ploy=np.zeros((X.shape[0],p),np.float32)
    m=X.shape[0]
    for i in range(m):
        for j in range(p):
            x_ploy[i,j]=X[i]**(j+1)
    return x_ploy


#归一化feature
def featureNormaliza(X):
    X_norm=np.array(X) #将X转化为numpy数组对象，才可以进行矩阵的运算
    #定义所需变量
    mu=np.zeros((1,X.shape[1]))
    sigma=np.zeros((1,X.shape[1]))
    mu=np.mean(X_norm,0)#求每一列的平均值 0代表列 1代表行
    sigma=np.std(X_norm,0) #求每一列的标准差
    for i in range(X.shape[1]):
        X_norm[:,i]=(X_norm[:,i]-mu[i])/sigma[i]  #归一化
    return X_norm,mu,sigma

#计算代价函数
def computerCost(theta,X,y,lamda):
    m = len(y)
    J = 0
    #theta=theta.reshape((X.shape[1],1))
    theta1=theta.copy()
    theta1[0]=0
    h=np.dot(X,theta)
    temp=np.dot(np.transpose(theta1),theta1)
    J = (np.dot((np.transpose(h - y)),(h - y))+temp*lamda)/(2*m)   # 计算代价J)
    #例如X为4*3矩阵，theta为3*1矩阵，X*theta转置后为1*4矩阵   再乘以4*1矩阵 行向量*列向量 即为代价的总和
    #print(J.shape)
    return J

#画二维图
def plot_X1_X2(X,y):
    plt.scatter(X,y)#画散点图
    plt.show()

#梯度下降计算theata
def gradientDescent(theta,X,y,lamda):
    m=len(y)
    #theta=theta.reshape((X.shape[1],1))
    #y=y.flat[0:m]
    h=np.dot(X,theta)
    #print(y)
    grad=np.zeros((theta.shape[0]))
    theta1=theta.copy()
    theta1[0]=0
    grad=np.dot(np.transpose(X),h-y)/m+lamda/m*theta1
    return grad

#画每次迭代变化的代价图
def plotJ(J_history,num_iters):
    x=np.arange(1,num_iters+1)
    plt.plot(x,J_history)
    plt.xlabel(u"迭代次数", fontproperties=font)  # 注意指定字体，要不然出现乱码问题
    plt.ylabel(u"代价值", fontproperties=font)
    plt.title(u"代价随迭代次数的变化", fontproperties=font)
    plt.show()

#预测
def predict(mu,sigma,theta,X):
    result=0
    X=ployfeatures(X,2)
    norm_predict = (X - mu) / sigma
    final_predict = np.hstack((np.ones((X.shape[0],1)), norm_predict))
    result = np.dot(final_predict, theta)  # 预测结果
    return result




if __name__ == "__main__":
    linearRegression()