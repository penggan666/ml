import numpy as np
from matplotlib import pyplot as plt
from scipy import io as spio
from sklearn.decomposition import pca
from sklearn.preprocessing import StandardScaler

'''二维降为一维'''
def PCA_2dto1d():
    data=spio.loadmat("data_pca.mat")
    X=data['X']
    #归一化数据
    scaler=StandardScaler()
    scaler.fit(X)
    x_train=scaler.transform(X)

    #数据降维
    K=1
    model=pca.PCA(n_components=K).fit(x_train)  #拟合数据
    Z=model.transform(x_train)  #tranform就会执行降维操作

    #数据恢复
    Ureduce=model.components_  #得到降维用的U
    print(Ureduce.shape)
    x_rec=np.dot(Z,Ureduce)
    plt=plot_data_2d(x_train,'bo')
    plot_data_2d(x_rec,'ro')
    for i in range(x_train.shape[0]):
        drawline(plt,x_train[i,:],x_rec[i,:],'--k')
    plt.show()

def drawline(plt,p1,p2,line_type):
    plt.plot(np.array([p1[0],p2[0]]),np.array([p1[1],p2[1]]),line_type)

'''图片降维并进行恢复'''
def PCA_face():
    image_data=spio.loadmat("faces.mat")
    X=image_data['X']
    display_imageData(X[0:100,:]) #显示100个最初图像
    scaler=StandardScaler()
    scaler.fit(X)
    x_train=scaler.transform(X)
    K=100
    model=pca.PCA(n_components=K).fit(x_train)
    Z=model.transform(x_train)
    Ureduce=model.components_
    x_rec=np.dot(Z,Ureduce)
    display_imageData(x_rec[0:100,:])

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

'''可视化二维数据'''
def plot_data_2d(X,marker):
    plt.plot(X[:,0],X[:,1],marker)
    return plt

if __name__=='__main__':
    PCA_2dto1d()
    #PCA_face()