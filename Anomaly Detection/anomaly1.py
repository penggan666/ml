from __future__ import print_function
import scipy.io as spio
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats #引入高斯分布


'''高斯分布主函数'''
def anomaly_test():
    data=spio.loadmat("ex8data1.mat")
    X=data['X']
    mu,sigma2=estimateGaussian(X)
    #p=oneUnitP(X,mu,sigma2)
    p=manyUnitP(X,mu,sigma2)
    '''选择异常点'''
    Xval=data['Xval']
    yval=data['yval']
    #pval=oneUnitP(Xval,mu,sigma2)  #计算CV上的概率密度值
    pval = manyUnitP(Xval, mu, sigma2)
    epsilon,F1=selectThreshold(yval,pval)  #在交叉验证集上选择最优的epsilon值和F1值
    print(u'在CV上得到的最好的epsilon是:%e'%epsilon)
    print(u'对用的F1值为：%f'%F1)
    outliers=np.where(p<epsilon)  #找到小于临界值的异常点，并作图
    plt=display_2d_data(X,'bx')
    plt.plot(X[outliers,0],X[outliers,1],'o',markeredgecolor='r',markerfacecolor='w',markersize=10.)
    plt.show()



'''画二维图像'''
def display_2d_data(X,marker):
    plt.plot(X[:,0],X[:,1],marker)
    plt.axis('square')
    return plt

'''求均值和方差'''
def estimateGaussian(X):
    m,n=X.shape
    mu=np.mean(X,axis=0)
    sigma2=np.var(X,axis=0)
    return mu,sigma2

'''单元高斯分布函数'''
def oneUnitP(x,mu,sigma2):
    # x is a new example:[m*n]
    m, n = x.shape
    p_list = []
    for j in range(m):
        p = 1
        for i in range(n):
            p = stats.norm.pdf(x[j, i], mu[i], np.sqrt(sigma2[i]))
        p_list.append(p)
    p_array = np.array(p_list).reshape(-1, 1)
    return p_array

'''多元高斯分布函数'''
def manyUnitP(X,mu,Sigma2):
    k = len(mu)
    if (Sigma2.shape[0]>1):
        Sigma2 = np.diag(Sigma2)
    '''多元高斯分布函数'''
    X = X-mu
    argu = (2*np.pi)**(-k/2)*np.linalg.det(Sigma2)**(-0.5)
    p = argu*np.exp(-0.5*np.sum(np.dot(X,np.linalg.inv(Sigma2))*X,axis=1))  # axis表示每行
    p=np.array(p).reshape(-1,1)
    return p


'''选择最优的epsilon，即使得F1score最大'''
def selectThreshold(y, pval):
    bestEpsilon = 0
    bestF1 = 0
    i=0
    stepSize = (np.max(pval) - np.min(pval)) / 1000

    for epsilon in np.arange(np.min(pval), np.max(pval), stepSize):
        predictions = (pval < epsilon)
        fp = np.sum((predictions == 1) & (y == 0))
        fn = np.sum((predictions == 0) & (y == 1))
        tp = np.sum((predictions == 1) & (y == 1))
        if tp + fp == 0:
            precision = 0
        else:
            precision = float(tp) / (tp + fp)  # note!!!!float!!!
        if tp + fn == 0:
            recall = 0
        else:
            recall = float(tp) / (tp + fn)

        if precision + recall == 0:
            F1 = 0
        else:
            F1 = 2.0 * precision * recall / (precision + recall)
        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon

    return bestEpsilon, bestF1


if __name__=='__main__':
    anomaly_test()