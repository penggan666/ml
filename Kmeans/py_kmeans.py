from __future__ import print_function
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from scipy import io as spio
from scipy import misc #图片操作
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)    # 解决windows环境下画图汉字乱码问题

'''聚类过程展示'''
def Kmeans():
    print(u'聚类过程展示\n')
    data=spio.loadmat('data.mat')
    X=data['X']
    K=3 #总类数
    #initial_centroids=np.array([[3,3],[6,2],[8,5]]) #初始化类中心
    X,mu,sigma=featureNormaliza(X)  #归一化
    initial_centroids=InitialCentroids(X,3)
    max_iters=10 #最大迭代次数
    runKmeans(X,initial_centroids,max_iters,True)

def featureNormaliza(X):
    X_norm=np.array(X)
    m,n=X_norm.shape
    mu=np.mean(X_norm,0)  #求每一列的均值
    sigma=np.std(X_norm,0)  #求每一列的标准差
    for i in range(n):  #对每一列进行归一化
        X_norm[:,i]=(X_norm[:,i]-mu[i])/sigma[i]
    return X_norm,mu,sigma

'''图像压缩'''
def Kmeans_photo():
    print(u'k-means压缩图片\n')
    img_data=misc.imread("bird_small.png")  #读取图片像素数据
    img_data=img_data/255.0  #像素值映射到0~1
    img_size=img_data.shape
    X=img_data.reshape(img_size[0]*img_size[1],3)  #调整为N*3的矩阵，N是所有像素点数
    K=8
    max_iter=5
    initial_centroids=InitialCentroids(X,K)
    centroids,idx=runKmeans(X,initial_centroids,max_iter,False)
    idx=findClosestCentroids(X,centroids)
    X_recovered=centroids[idx,:]  #将k个中心值赋给X_recovered以此来实现压缩图像
    X_recovered=X_recovered.reshape(img_size[0],img_size[1],3)
    plt.subplot(1,2,1)
    plt.imshow(img_data)
    plt.title(u'原图',fontproperties=font)
    plt.subplot(1,2,2)
    plt.imshow(X_recovered)
    plt.title(u'压缩图',fontproperties=font)
    plt.show()



'''找到每条数据距离哪个类中心最近'''
def findClosestCentroids(X,initial_centroids):
    m=X.shape[0] #数据条数
    k=initial_centroids.shape[0] #类数
    dis=np.zeros((m,k)) #存储每个点分别到k个类中心的距离
    idx=np.zeros((m,1)) #存储每个点属于哪个类
    '''计算每个点到每个类中心的距离'''
    for i in range(m):
        for j in range(k):
            dis[i,j]=np.dot((X[i,:]-initial_centroids[j,:]).reshape(1,-1),(X[i,:]-initial_centroids[j,:]).reshape(-1,1))
    '''找到每一行最小的值的下标'''
    dummy,idx=np.where(dis==np.min(dis,axis=1).reshape(-1,1))
    total=0
    for i in range(m):
        total=total+dis[i,idx[i]]
    print(u'总代价为:%f'%(total))
    return idx[0:dis.shape[0]]




def runKmeans(X,initial_centroids,max_iters,plot_process):
    m,n=X.shape  #数据条数和维度
    k=initial_centroids.shape[0] #类数
    centroids=initial_centroids #记录当前类中心
    pre_centroids=centroids #记录上一次类中心
    idx=np.zeros((m,1))  #记录每一条数据属于哪个类

    for i in range(max_iters):
        print(u'迭代计算次数：%d'%(i+1))
        idx=findClosestCentroids(X,centroids)
        if plot_process:
            plt=plotProcess(X,centroids,pre_centroids)
            pre_centroids=centroids #重置
        centroids=computerCentroids(X,idx,k)#重新计算类中心
    if plot_process: #显示最终的绘制结果
        plt.show()
    return centroids,idx #返回聚类中心和分类结果

'''计算类中心'''
def computerCentroids(X,idx,K):
    n=X.shape[1]
    centroids=np.zeros((K,n))
    for i in range(K):
        centroids[i,:]=np.mean(X[np.ravel(idx==i),:],axis=0).reshape(1,-1)
    return centroids

'''画出聚类中心的移动过程'''
def plotProcess(X,centroids,previous_centroids):
    plt.scatter(X[:,0],X[:,1]) #原数据的散点图
    plt.plot(previous_centroids[:,0],previous_centroids[:,1],'rx') #上一次聚类中心
    plt.plot(centroids[:,0],centroids[:,1],'rx')  #当前聚类中心
    for j in range(centroids.shape[0]): #遍历每个类，画类中心的移动直线
        p1=centroids[j,:]
        p2=previous_centroids[j,:]
        plt.plot([p1[0],p2[0]],[p1[1],p2[1]],"->",linewidth=2.0)
    return plt

'''初始化类中心，随机选取k个点作为聚类中心'''
def InitialCentroids(X,K):
    m=X.shape[0]
    m_arr=np.arange(0,m)  #生成0~m-1
    centroids=np.zeros((K,X.shape[1]))
    np.random.shuffle(m_arr)  #打乱m_arr顺序
    rand_indices=m_arr[:K]  #取前K个
    centroids=X[rand_indices,:]
    return centroids

if __name__=='__main__':
    Kmeans()
    #Kmeans_photo()