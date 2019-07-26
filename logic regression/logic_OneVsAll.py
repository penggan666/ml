from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
from scipy import optimize
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)    # 解决windows环境下画图汉字乱码问题

def logisticRegression_OneVsAll():
    data=spio.loadmat("data_digits.mat")
    X=data['X'] #获取X数据 每一行对应一个数字20*20px   数据大小为5000行400列
    y=data['y'] #数据大小为5000行 1列
    m,n=X.shape #m是行数 n是列数
    num_labels=10 #数字个数 0-9
    ##随机显示几行数据
    rand_indices=[t for t in [np.random.randint(x-x,m) for x in range(100)]] #生成100个0-m的随机数
    display_data(X[rand_indices,:]) #显示100个数字
    lamda=0.1
    all_theta=oneVsAll(X,y,num_labels,lamda)
    print(all_theta.shape)
    p = predict_OneVsAll(all_theta, X)  # 预测
    # 将预测结果和真实结果保存到文件中
    res = np.hstack((p,y.reshape(-1,1)))
    np.savetxt("predict.csv", res, delimiter=',')

    print(u"预测准确度为：%f%%" % np.mean(np.float64(p == y.reshape(-1, 1)) * 100))

def oneVsAll(X,y,num_labels,lamda):
    #初始化变量
    m,n=X.shape
    all_theta=np.zeros((n+1,num_labels)) #每一列对应相应分类的theta，共10列
    X=np.hstack((np.ones((m,1)),X)) #X前补上一列1的偏置bias
    class_y=np.zeros((m,num_labels)) #数据的y对应0-9，需要映射为0/1的关系
    initial_theta=np.zeros((n+1,1))

    #映射y
    for i in range(num_labels):
        class_y[:,i]=np.int32(y==i).reshape(1,-1) #注意reshape(1,-1)才可以赋值

    #遍历每个分类，计算对应的theta值
    for i in range(num_labels):
        result=optimize.fmin_bfgs(costFunction,initial_theta,fprime=gradient,args=(X,class_y[:,i],lamda))
        all_theta[:,i]=result.reshape(1,-1)
    return np.transpose(all_theta)

#代价函数
def costFunction(initial_theta,X,y,initial_lamda):
    m=len(y)
    J=0

    h=sigmoid(np.dot(X,initial_theta))  #计算hz
    theta1=initial_theta.copy()
    theta1[0]=0
    temp=np.dot(np.transpose(theta1),theta1)
    J=(-np.dot(np.transpose(y),np.log(h))-np.dot(np.transpose(1-y),np.log(1-h))+temp*initial_lamda/2)/m
    return J

#计算梯度
def gradient(initial_theta,X,y,initial_lamda):
    m=len(y)
    gard=np.zeros((initial_theta.shape[0]))
    h=sigmoid(np.dot(X,initial_theta))
    theta1=initial_theta.copy()
    theta1[0]=0

    grad=np.dot(np.transpose(X),h-y)/m+initial_lamda/m*theta1
    return grad

#h函数
def sigmoid(z):
    h=np.zeros((len(z),1)) #初始化，与z的长度一致

    h=1.0/(1.0+np.exp(-z))
    return h

#显示100个数字
def display_data(imgData):
    sum=0
    #显示100个数（若是一个一个绘制会非常慢，可以将要画的数字整理好，放到一个矩阵中，希纳是这个矩阵即可
    #初始化一个二维数组
    #将每行的数据调整成图像的矩阵，放进二维数组
    #显示即可
    pad=1
    display_array=-np.ones((pad+10*(20+pad),pad+10*(20+pad)))
    for i in range(10):
        for j in range(10):
            display_array[pad + i * (20 + pad):pad + i * (20 + pad) + 20,
            pad + j * (20 + pad):pad + j * (20 + pad) + 20] = (
            imgData[sum, :].reshape(20, 20, order="F"))  # order=F指定以列优先，在matlab中是这样的，python中需要指定，默认以行
            sum += 1
    plt.imshow(display_array,cmap='gray')   #显示灰度图像
    plt.axis('off')
    plt.show()

#预测
def predict_OneVsAll(all_theta,X):
    m=X.shape[0]
    num_labels=all_theta.shape[1]
    p=np.zeros((m,1))
    X=np.hstack((np.ones((m,1)),X))
    h=sigmoid(np.dot(X,np.transpose(all_theta)))
    #返回h中每一行最大值所在的列号
    #np.max(h,axis=1)返回h中每一行的最大值
    #最后where找到最大概率所在的列号（列号即是对应的数字）
    p = np.array(np.where(h[0,:] == np.max(h, axis=1)[0]))
    for i in np.arange(1, m):
        t = np.array(np.where(h[i,:] == np.max(h, axis=1)[i]))
        p = np.vstack((p,t))
    return p

if __name__=="__main__":
    logisticRegression_OneVsAll()