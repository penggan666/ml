from __future__ import print_function
from numpy import *

def loadSimpData():
    '''
    测试数据
    :return:
     dataArr feature对应的数据集
     labelArr frature对应的分类标签
    '''
    dataArr = array([[1., 2.1], [2., 1.1], [1.3, 1.], [1., 1.], [2., 1.]])
    labelArr = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataArr, labelArr

def loadDataSet(fileName):
    numFeat=len(open(fileName).readline().split('\t'))
    dataArr=[]
    labelArr=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataArr.append(lineArr)
        labelArr.append(float(curLine[-1]))
    return dataArr,labelArr

def stumpClassify(dataMat,dimen,threshVal,threshIneq):
    '''
    将数据集，按照feature列的value进行二分法切分比较来赋值分类
    :param dataMat: Matrix数据集
    :param dimen: 特征列
    :param threshVal: 特征列要比较的值
    :param threshIneq: 
    :return: 结果集
    '''
    #默认都是1
    retArray=ones((shape(dataMat)[0],1))
    if threshIneq=='lt': #表示修改左边的值
        retArray[dataMat[:,dimen]<=threshVal]=-1.0
    else:
        retArray[dataMat[:, dimen] > threshVal] = -1.0
    return retArray

def buildStump(dataArr,labelArr,D):
    '''
    得到决策树模型
    :param dataArr:特征标签集合 
    :param labelArr: 分类标签集合
    :param D: 最初的样本的所有特征权值集合
    :return: 
    bestStump:最优的分类器模型
    minError：错误率
    bestClasEst：训练后的结果集
    '''
    #转换数据
    dataMat=mat(dataArr)
    labelMat=mat(labelArr).T
    m,n=shape(dataMat)
    #初始化数据
    numSteps=10.0
    bestStump={}
    bestClassEst=mat(zeros((m,1)))
    #初始化最小误差为无穷大
    minError=inf
    #循环所有的feature列，将列切分成若干份，每一段以最左边的点作为分类节点
    for i in range(n):
        rangeMin=dataMat[:,i].min()
        rangeMax=dataMat[:,i].max()
        #计算每一份的元素个数
        stepSize=(rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt','gt']:
                #如果是-1，那么得到rangeMin-stepSize;如果是numSteps，那么得到rangeMa'x