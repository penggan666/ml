from __future__ import print_function
from numpy import *

'''案例一 判断侮辱性言论'''
def loadDataSet():
    """
    创建数据集
    :return: 单词列表postingList, 所属类别classVec
    """
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'], #[0,0,1,1,1......]
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec

#获取所有单词的集合,返回值为所有单词的集合
def createVocabList(dataSet):
    vocabSet=set([])
    for document in dataSet:
        #操作符|用于求两个集合的并集
        vocabSet=vocabSet|set(document)
    return list(vocabSet)

#遍历查看该单词是否出现，出现该单词则将该单词置1
def setOfWords2Vec(vocabList,inputSet):
    '''
    :param vocabList:所有单词集合列表 
    :param inputSet: 输入数据集
    :return: 匹配列表【0，1，0，1。。。】，其中0和1表示词汇表中的单词是否出现在输入集中
    '''
    #创建一个和词汇表等长的向量，并将其元素都设置为0
    returnVec=[0]*len(vocabList)
    #遍历文档中的所有单词
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else:
            print("the word %s is not in my Vocabulary" %word)
    return returnVec

def trainNB0(trainMatrix,trainCategory):
    '''
    :param trainMatrix:文件单词矩阵
    :param trainCategory: 文件对应的类别
    :return: 
    '''
    #总文件数
    numTrainDocs=len(trainMatrix)
    #总单词数
    numWords=len(trainMatrix[0])
    #侮辱性文件出现的概率
    pAbusive=sum(trainCategory)/float(numTrainDocs)
    #构造单词出现次数列表
    #避免单词列表中的任何一个单词为0，而导致最后的乘积为0，所以将每个单词出现次数初始化为1
    p0Num=ones(numWords) #正常的统计
    p1Num=ones(numWords) #侮辱的统计
    #整个数据集单词出现总数
    #2.0根据样本/实际调查结果调整分母的值（主要是为了避免分母为0）
    p0Denom=2.0 #正常的统计
    p1Denom=2.0 #侮辱的统计
    print(trainMatrix[1])
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            #累加辱骂词的频次
            p1Num+=trainMatrix[i]
            p1Denom+=sum(trainMatrix[i])
        else:
            p0Num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])
    # 类别1，即侮辱性文档的[log(P(F1|C1)),log(P(F2|C1)),log(P(F3|C1)),log(P(F4|C1)),log(P(F5|C1))....]列表
    p1Vect = log(p1Num / p1Denom)
    # 类别0，即正常文档的[log(P(F1|C0)),log(P(F2|C0)),log(P(F3|C0)),log(P(F4|C0)),log(P(F5|C0))....]列表
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    """
        使用算法：
            # 将乘法转换为加法
            乘法：P(C|F1F2...Fn) = P(F1F2...Fn|C)P(C)/P(F1F2...Fn)
            加法：P(F1|C)*P(F2|C)....P(Fn|C)P(C) -> log(P(F1|C))+log(P(F2|C))+....+log(P(Fn|C))+log(P(C))
        :param vec2Classify: 待测数据[0,1,1,1,1...]，即要分类的向量
        :param p0Vec: 类别0，即正常文档的[log(P(F1|C0)),log(P(F2|C0)),log(P(F3|C0)),log(P(F4|C0)),log(P(F5|C0))....]列表
        :param p1Vec: 类别1，即侮辱性文档的[log(P(F1|C1)),log(P(F2|C1)),log(P(F3|C1)),log(P(F4|C1)),log(P(F5|C1))....]列表
        :param pClass1: 类别1，侮辱性文件的出现概率
        :return: 类别1 or 0
        """
    # 计算公式  log(P(F1|C))+log(P(F2|C))+....+log(P(Fn|C))+log(P(C))
    # 使用 NumPy 数组来计算两个向量相乘的结果，这里的相乘是指对应元素相乘，即先将两个向量中的第一个元素相乘，然后将第2个元素相乘，以此类推。
    # 我的理解是：这里的 vec2Classify * p1Vec 的意思就是将每个词与其对应的概率相关联起来
    # 可以理解为 1.单词在词汇表中的条件下，文件是good 类别的概率 也可以理解为 2.在整个空间下，文件既在词汇表中又是good类别的概率
    p1=sum(vec2Classify*p1Vec)+log(pClass1)
    p0=sum(vec2Classify*p0Vec)+log(1.0-pClass1)
    if p1>p0:
        return 1
    else:
        return 0

def testingNB():
    '''
    测试朴素贝叶斯算法
    :return: 
    '''
    #加载数据集
    listOPosts,listClasses=loadDataSet()
    #创建单词集合
    myVocabList=createVocabList(listOPosts)
    #计算单词是否出现并创建数据矩阵
    trainMat=[]
    for postinDoc in listOPosts:
        #返回m*len(myVocabList)的矩阵，记录的都是0，1信息
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    #训练数据
    p0V,p1V,pAb=trainNB0(array(trainMat),array(listClasses))
    #测试数据
    testEntry=['love','my','dalmation']
    thisDoc=array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))


'''案例2 过滤垃圾邮件'''
def textParse(bigString):
    '''
    接收一个大字符串并将其解析为字符串列表
    :param bigString: 大字符串
    :return: 去掉少于2个字符的字符串，并将所有字符串转换为小写，返回字符串列表
    '''
    import re
    #使用正则表达式来切分句子，其中分隔符是除单词，数字外的任意字符串
    listOfTokens=re.split(r'\W*',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok)>2]

def spamTest():
    '''
    对贝叶斯垃圾邮件分类器进行自动化处理
    :return: 对测试集中的每封邮件进行分类，若邮件分类错误，则错误数+1，最后返回总的错误比
    '''
    docList=[]
    classList=[]
    fullText=[]
    for i in range(1,26):
        #切分、解析数据，并归类为1类别
        wordList=textParse(open('spam/%d.txt' %i).read())
        docList.append(wordList)
        classList.append(1)
        #切分，解析数据，并归类为0类别
        '''
        extend和append的区别：
        extend接受一个参数，这个参数总是一个list，并且把这个list中的每个元素添加到原list中
        append接受一个参数，这个参数可以是任何数据类型，并且简单地追加到list的尾部
        '''
        wordList=textParse(open('ham/%d.txt' %i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    #创建词汇表
    vocabList=createVocabList(docList)
    trainingSet=list(range(50))
    testSet=[]
    #随机取10个邮件用来测试
    for i in range(10):
        #random.uniform(x,y)随机生成一个范围在x~y的数
        randIndex=int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat=[]
    trainClasses=[]
    #构建训练集
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex])) #邮件内容
        temp=classList[docIndex]
        trainClasses.append(temp)  #邮件分类
    p0V,p1V,pSam=trainNB0(array(trainMat),array(trainClasses))
    errorCount=0
    #测试分类准确率
    for docIndex in testSet:
        wordVector=setOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSam)!=classList[docIndex]:
            errorCount+=1
    print('the errorCount is:',errorCount)
    print('the testSet length is:',len(testSet))
    print('the error rate is:',float(errorCount)/len(testSet))



if __name__=='__main__':
    #testingNB()
    spamTest()
