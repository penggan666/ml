import numpy as np
from matplotlib import pyplot as plt
from scipy import io as spio

'''二维降为一维'''
def PCA_2dto1d():
    data=spio.loadmat("data_pca.mat")
    X=data['X']
    X_copy=X.copy()
    m=X.shape[0]
    #X_norm,mu,sigma=featureNormalize(X_copy) #特征值归一化
    X_norm=X_copy
    Sigma=np.dot(np.transpose(X_norm),X_norm)/m #求Sigma
    U,S,V=np.linalg.svd(Sigma) #对Sigma进行奇异值分解
    print(U.shape)
    K=1 #定义要降的维度
    Z,Ureduce=projectData(X_norm,U,K) #执行降维操作
    X_rec=np.dot(Z,np.transpose(Ureduce)) #恢复数据
    plt=plot_data_2d(X_norm,'bo')
    plot_data_2d(X_rec,'ro')
    for i in range(m):
        drawline(plt,X_norm[i,:],X_rec[i,:],'--k')
    plt.show()

'''压缩图像'''
def pca_photo():
    data=spio.loadmat("faces.mat")
    X=data['X']
    display_img=X[0:100,:]
    print(display_img.shape[1])
    display_imageData(display_img)
    #X_norm,mu,sigma=featureNormalize(display_img)
    X_norm=display_img
    Sigma=np.dot(np.transpose(X_norm),X_norm)/X_norm.shape[0]
    U,S,V=np.linalg.svd(Sigma)
    K=100
    Z,Ureduced=projectData(X_norm,U,K)
    X_rec=np.dot(Z,np.transpose(Ureduced))
    display_imageData(X_rec)

'''特征值归一化'''
def featureNormalize(X):
    X_norm=np.array(X)
    m,n=X_norm.shape
    mu=np.mean(X_norm,axis=0)
    sigma=np.std(X_norm,axis=0)
    for i in range(n):
        X_norm[:,i]=(X_norm[:,i]-mu[i])/sigma[i]
    return X_norm,mu,sigma

# 显示图片
def display_imageData(imgData):
    sum = 0
    '''
    显示100个数（若是一个一个绘制将会非常慢，可以将要画的图片整理好，放到一个矩阵中，显示这个矩阵即可）
    - 初始化一个二维数组
    - 将每行的数据调整成图像的矩阵，放进二维数组
    - 显示即可
    '''
    m, n = imgData.shape
    width = np.int32(np.round(np.sqrt(n)))
    height = np.int32(n / width);
    rows_count = np.int32(np.floor(np.sqrt(m)))
    cols_count = np.int32(np.ceil(m / rows_count))
    pad = 1
    display_array = -np.ones((pad + rows_count * (height + pad), pad + cols_count * (width + pad)))
    for i in range(rows_count):
        for j in range(cols_count):
            max_val = np.max(np.abs(imgData[sum, :]))
            display_array[pad + i * (height + pad):pad + i * (height + pad) + height,
            pad + j * (width + pad):pad + j * (width + pad) + width] = imgData[sum, :].reshape(height, width,
                                                                                               order="F") / max_val  # order=F指定以列优先，在matlab中是这样的，python中需要指定，默认以行
            sum += 1

    plt.imshow(display_array, cmap='gray')  # 显示灰度图像
    plt.axis('off')
    plt.show()

#进行降维得到Z
def projectData(X_norm,U,K):
    Z=np.zeros((X_norm.shape[0],K))
    Ureduce=U[:,0:K] #取U的前K列
    Z=np.dot(X_norm,Ureduce)
    return Z,Ureduce

#可视化二维数据
def plot_data_2d(X,marker):
    plt.plot(X[:,0],X[:,1],marker)
    return plt

#画一条线
def drawline(plt,p1,p2,line_type):
    plt.plot(np.array([p1[0],p2[0]]),np.array([p1[1],p2[1]]),line_type)

if __name__=='__main__':
    PCA_2dto1d()
    pca_photo()
