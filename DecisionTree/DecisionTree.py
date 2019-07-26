from __future__ import print_function
print(__doc__)
import operator
import matplotlib.pyplot as plt
from math import log
from collections import Counter
#Counter是一个有助于hashtable对象计数的dict子类。它是一个无序的集合，其中hashtable对象的元素存储为字典的键，它们的计数存储为字典的值
#计数可以为任意整数，包括零和负数

'''画决策树函数'''
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    # 根节点开始遍历
    for key in secondDict.keys():
        # 判断子节点是否为dict, 不是+1
        if type(secondDict[key]) is dict:
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    # 根节点开始遍历
    for key in secondDict.keys():
        # 判断子节点是不是dict, 求分枝的深度
        if type(secondDict[key]) is dict:
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        # 记录最大的分支深度
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction', xytext=centerPt,
                            textcoords='axes fraction', va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


def plotTree(myTree, parentPt, nodeTxt):
    # 获取叶子节点的数量
    numLeafs = getNumLeafs(myTree)
    # 获取树的深度
    # depth = getTreeDepth(myTree)

    # 找出第1个中心点的位置，然后与 parentPt定点进行划线
    # x坐标为 (numLeafs-1.)/plotTree.totalW/2+1./plotTree.totalW，化简如下
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    # print cntrPt
    # 并打印输入对应的文字
    plotMidText(cntrPt, parentPt, nodeTxt)

    firstStr = list(myTree.keys())[0]
    # 可视化Node分支点；第一次调用plotTree时，cntrPt与parentPt相同
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    # 根节点的值
    secondDict = myTree[firstStr]
    # y值 = 最高点-层数的高度[第二个节点位置]；1.0相当于树的高度
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        # 判断该节点是否是Node节点
        if type(secondDict[key]) is dict:
            # 如果是就递归调用[recursion]
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            # 如果不是，就在原来节点一半的地方找到节点的坐标
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            # 可视化该节点位置
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            # 并打印输入对应的文字
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def createPlot(inTree):
    # 创建一个figure的模版
    fig = plt.figure(1, facecolor='green')
    fig.clf()

    axprops = dict(xticks=[], yticks=[])
    # 表示创建一个1行，1列的图，createPlot.ax1 为第 1 个子图，
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)

    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    # 半个节点的长度；xOff表示当前plotTree未遍历到的最左的叶节点的左边一个叶节点的x坐标
    # 所有叶节点中，最左的叶节点的x坐标是0.5/plotTree.totalW（因为totalW个叶节点在x轴方向是平均分布在[0, 1]区间上的）
    # 因此，xOff的初始值应该是 0.5/plotTree.totalW-相邻两个叶节点的x轴方向距离
    plotTree.xOff = -0.5 / plotTree.totalW
    # 根节点的y坐标为1.0，树的最低点y坐标为0
    plotTree.yOff = 1.0
    # 第二个参数是根节点的坐标
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()
'''画决策树结束-------------------------'''




#创建基础数据集
def createDataSet():
    dataset=[[1,1,'yes'],
             [1,1,'yes'],
             [1,0,'no'],
             [0,1,'no'],
             [0,1,'no']]
    #标签 露出水面 脚蹼
    labels=['no surfacing','flippers']
    return dataset,labels

#计算给定数据集的香农熵,返回值为每一组feature下的某个分类下，香农熵的信息期望
def calcshannonEnt(dataSet):
    '''计算香农熵的第一种实现方式'''
    numEntries=len(dataSet)
    #计算分类标签label出现的次数
    labelCounts={}
    for featVec in dataSet:
        # 将当前实例的标签存储，即每一行数据的最后一个数据代编是标签
        currentLabel=featVec[-1]
        #为所有可能的分类创建字典，如果当前的键值不存在，则扩展该字典，每个键值都记录了当前类别出现的次数
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1
    #对于lable标签的占比，求出label标签的香农熵
    shannonEnt=0.0
    for key in labelCounts:
        #使用所有类标签的发生频率计算类别出现的频率
        prob=float(labelCounts[key]/numEntries)
        #计算香农熵
        shannonEnt-=prob*log(prob,2)

    '''计算香农熵的第二种实现方式
    #计算标签出现的次数
    label_count=Counter(data[-1] for data in dataSet)
    #计算概率
    probs=[p[1]/len(dataSet) for p in label_count.items()]
    #计算香农熵
    shannonEnt=sum([-p*log(p,2) for p in probs])
    '''
    return shannonEnt

#通过遍历dataset数据集，求出index对应的column列的值为value的行
def spiltDataSet(dataSet,index,value):
    retDataSet=[]
    for featVec in dataSet:
        #index列为value的数据集【该数据集需要排除index列】
        #判断index列的值是否为value
        if featVec[index]==value:
            reduceFeatVec=featVec[:index]
            reduceFeatVec.extend(featVec[index+1:])  #跳过index这一列
            retDataSet.append(reduceFeatVec)
    return retDataSet

#选择最好的特征,返回最优的特征列
def chooseBestFeatureToSplit(dataSet):
    numFeatures=len(dataSet[0])-1 #得到有多少特征
    #标签的信息熵
    baseEntropy=calcshannonEnt(dataSet)
    #最优的信息增益值和最优的Feature编号
    bestInfoGain,bestFeature=0.0,-1
    for i in range(numFeatures):
        #获取每个一实例的第i+1个feature，组成list集合
        featList=[example[i] for example in dataSet]
        #使用set剔重
        uniqueVals=set(featList)
        #临时信息熵
        newEntropy=0.0
        # 遍历某一列的value集合，计算该列的信息熵
        # 遍历当前特征中的所有唯一属性值，对每个唯一属性值划分一次数据集，计算数据集的新熵值，并对所有唯一特征值得到的熵求和。
        for value in uniqueVals:
            subDataSet=spiltDataSet(dataSet,i,value)
            prob=len(subDataSet)/float(len(dataSet))
            newEntropy+=prob*calcshannonEnt(subDataSet)
            # gain[信息增益]: 划分数据集前后的信息变化， 获取信息熵最大的值
            # 信息增益是熵的减少或者是数据无序度的减少。最后，比较所有特征中的信息增益，返回最好特征划分的索引值。
            infoGain = baseEntropy - newEntropy
            print('infoGain=', infoGain, 'bestFeature=', i, baseEntropy, newEntropy)
            if (infoGain > bestInfoGain):
                bestInfoGain = infoGain
                bestFeature = i
    return bestFeature

#选择出现次数最多的一个结果,输入为lable列的集合，输出为最优的特征列
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    #倒序排列，取出出现次数最多的那个结果
    sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=true)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # 如果数据集的最后一列的第一个值出现的次数=整个集合的数量，也就说只有一个类别，就只直接返回结果就行
    # 第一个停止条件：所有的类标签完全相同，则直接返回该类标签。
    # count() 函数是统计括号中的值在list中出现的次数
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 如果数据集只有1列，那么最初出现label次数最多的一类，作为结果
    # 第二个停止条件：使用完了所有特征，仍然不能将数据集划分成仅包含唯一类别的分组。
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    # 选择最优的列，得到最优列对应的label含义
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 获取label的名称
    bestFeatLabel = labels[bestFeat]
    # 初始化myTree
    myTree = {bestFeatLabel: {}}
    # 注：labels列表是可变对象，在PYTHON函数中作为参数时传址引用，能够被全局修改
    # 所以这行代码导致函数外的同名变量被删除了元素，造成例句无法执行，提示'no surfacing' is not in list
    del(labels[bestFeat])
    # 取出最优列，然后它的branch做分类
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # 求出剩余的标签label
        subLabels = labels[:]
        # 遍历当前选择特征包含的所有属性值，在每个数据集划分上递归调用函数createTree()
        myTree[bestFeatLabel][value] = createTree(spiltDataSet(dataSet, bestFeat, value), subLabels)
        # print 'myTree', value, myTree
    return myTree

def classify(inputTree, featLabels, testVec):
    """classify(给输入的节点，进行分类)
    Args:
        inputTree  决策树模型
        featLabels Feature标签对应的名称
        testVec    测试输入的数据
    Returns:
        classLabel 分类的结果值，需要映射label才能知道名称
    """
    # 获取tree的根节点对应的key值
    firstStr = list(inputTree.keys())[0]
    # 通过key得到根节点对应的value
    secondDict = inputTree[firstStr]
    # 判断根节点名称获取根节点在label中的先后顺序，这样就知道输入的testVec怎么开始对照树来做分类
    featIndex = featLabels.index(firstStr)
    # 测试数据，找到根节点对应的label位置，也就知道从输入的数据的第几位来开始分类
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    print('+++', firstStr, 'xxx', secondDict, '---', key, '>>>', valueOfFeat)
    # 判断分枝是否结束: 判断valueOfFeat是否是dict类型
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel


def fishTest():
    # 1.创建数据和结果标签
    myDat, labels = createDataSet()
    # print myDat, labels

    # 计算label分类标签的香农熵
    # calcShannonEnt(myDat)

    # # 求第0列 为 1/0的列的数据集【排除第0列】
    # print '1---', splitDataSet(myDat, 0, 1)
    # print '0---', splitDataSet(myDat, 0, 0)

    # # 计算最好的信息增益的列
    # print chooseBestFeatureToSplit(myDat)

    import copy
    myTree = createTree(myDat, copy.deepcopy(labels))
    print(myTree)
    # [1, 1]表示要取的分支上的节点位置，对应的结果值
    print(classify(myTree, labels, [1, 1]))

    # 获得树的高度
    print(get_tree_height(myTree))

    # 画图可视化展现
    createPlot(myTree)

def get_tree_height(tree):
    """
     Desc:
        递归获得决策树的高度
    Args:
        tree
    Returns:
        树高
    """

    if not isinstance(tree, dict):
        return 1

    child_trees = list(tree.values())[0].values()

    # 遍历子树, 获得子树的最大高度
    max_height = 0
    for child_tree in child_trees:
        child_tree_height = get_tree_height(child_tree)

        if child_tree_height > max_height:
            max_height = child_tree_height

    return max_height + 1

def ContactLensesTest():
    """
    Desc:
        预测隐形眼镜的测试代码
    Args:
        none
    Returns:
        none
    """

    # 加载隐形眼镜相关的 文本文件 数据
    fr = open('lenses.txt')
    # 解析数据，获得 features 数据
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    # 得到数据的对应的 Labels
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    # 使用上面的创建决策树的代码，构造预测隐形眼镜的决策树
    lensesTree = createTree(lenses, lensesLabels)
    print(lensesTree)
    # 画图可视化展现
    createPlot(lensesTree)


if __name__=='__main__':
    #fishTest()
    ContactLensesTest()