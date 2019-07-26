from __future__ import print_function
import scipy.io as spio
import scipy.linalg as la
import scipy.optimize as op
import numpy as np
import matplotlib.pyplot as plt

def Recommender_test():
    data=spio.loadmat("ex8_movies")
    Y=data['Y']   #Y是不同电影的不同用户的评分，行数为电影数目，列数为用户数目
    R=data['R']   #R是二进制矩阵，象征某用户是否给某电影进行了打分
    movieList=loadMovieList()
    my_ratings=np.zeros((1682,))
    my_ratings[0] = 4
    my_ratings[97] = 2
    my_ratings[6] = 3
    my_ratings[11] = 5
    my_ratings[53] = 4
    my_ratings[63] = 5
    my_ratings[65] = 3
    my_ratings[68] = 5
    my_ratings[182] = 4
    my_ratings[225] = 5
    my_ratings[354] = 5
    print('New user ratings:')
    for i in range(np.size(my_ratings, 0)):
        if my_ratings[i] > 0:
            print('Rated %d for %s' % (my_ratings[i], movieList[i]))
    Y = np.c_[my_ratings.reshape((np.size(my_ratings, 0), 1)), Y]
    R = np.c_[(my_ratings != 0).reshape((np.size(my_ratings, 0), 1)), R]
    Y_norm,Ymean=normalizeRating(Y,R)
    num_users=np.size(Y,1)
    num_movies=np.size(Y,0)
    num_features=10
    #随机初始化X和Theta
    X=np.random.randn(num_movies,num_features)
    Theta=np.random.randn(num_users,num_features)
    init_params=np.hstack((X.flatten(),Theta.flatten()))
    lamb=10
    theta=op.fmin_cg(costFunction,init_params,fprime=GradFunction,
                     args=(Y, R, num_users, num_movies, num_features, lamb), maxiter=100)
    X = np.reshape(theta[0: num_movies * num_features], (num_movies, num_features))
    Theta = np.reshape(theta[num_movies * num_features:], (num_users, num_features))
    '''为你推荐电影'''
    p=X.dot(Theta.T)
    my_pred=p[:,0]+Ymean  #要加上均值来恢复源数据
    print(my_pred)
    ix=np.argsort(my_pred)[::-1] #将预测评分从大到小排列
    print('Top recommendaeions for you:')
    for i in range(10):
        j=ix[i]
        print('Predicting rating %.1f for movie %s'%(my_pred[j],movieList[j]))
    print("end!")

'''计算代价函数'''
def costFunction(params,Y,R,num_users,num_movies,num_features,lam):
    #还原X和theta
    X = np.reshape(params[0: num_movies*num_features], (num_movies, num_features))
    Theta = np.reshape(params[num_movies*num_features:], (num_users, num_features))
    J = 1/2*np.sum(R*(X.dot(Theta.T)-Y)**2)+lam/2*(np.sum(Theta**2)+np.sum(X**2))
    return J

'''计算梯度'''
def GradFunction(params,Y,R,num_users,num_movies,num_features,lam):
    #还原X和theta
    X = np.reshape(params[0: num_movies * num_features], (num_movies, num_features))
    Theta = np.reshape(params[num_movies * num_features:], (num_users, num_features))

    X_grad = np.zeros(X.shape)
    Theta_grad = np.zeros(Theta.shape)
    for i in range(np.size(X, 0)):
        idx = R[i, :] == 1  #找到该电影被哪些用户评过分
        X_grad[i, :] = (X[i, :].dot(Theta[idx, :].T)-Y[i, idx]).dot(Theta[idx, :])+lam*X[i, :]
    for j in range(np.size(Theta, 0)):
        jdx = R[:, j] == 1  #找到该用户给哪些电影评过分
        Theta_grad[j, :] = (Theta[j, :].dot(X[jdx, :].T)-Y[jdx, j].T).dot(X[jdx, :])+lam*Theta[j, :]
    grad = np.hstack((X_grad.flatten(), Theta_grad.flatten()))
    return grad

'''加载电影数据'''
def loadMovieList():
    movieList = [line.split(' ', 1)[1] for line in open('movie_ids.txt', encoding='utf8')]
    return movieList

'''归一化'''
def normalizeRating(Y, R):
    m, n = Y.shape
    Ymean = np.zeros((m,))
    Ynorm = np.zeros(Y.shape)
    for i in range(m):
        idx = R[i, :] == 1
        Ymean[i] = np.mean(Y[i, idx])
        Ynorm[i, idx] = Y[i, idx]-Ymean[i]
    return Ynorm, Ymean

if __name__=='__main__':
    Recommender_test()