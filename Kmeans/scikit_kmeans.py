import numpy as np
from scipy import io as spio
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.image as mpimg


def kmenas():
    data = spio.loadmat("data.mat")
    X = data['X']
    model = KMeans(n_clusters=3).fit(X)  # n_clusters指定3类，拟合数据
    centroids = model.cluster_centers_  # 聚类中心

    plt.scatter(X[:, 0], X[:, 1])  # 原数据的散点图
    plt.plot(centroids[:, 0], centroids[:, 1], 'r^', markersize=10)  # 聚类中心
    plt.show()

'''图像压缩'''
def Kmeans_photo():
    pixel=mpimg.imread('bird_small.png')
    pixel=pixel.reshape((128*128,3))
    pixel=pixel/2255.0
    model=KMeans(n_clusters=8).fit(pixel)
    newpixel=[]
    '''model.labels_即是每个点被分配给哪个集合'''
    for i in model.labels_:
        newpixel.append(list(model.cluster_centers_[i,:]))
    newpixel=np.array(newpixel)
    newpixel=newpixel.reshape((128,128,3))
    plt.imshow(newpixel)
    plt.show()

if __name__ == "__main__":
    kmenas()
    Kmeans_photo()