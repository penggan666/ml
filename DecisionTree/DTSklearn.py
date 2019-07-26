from __future__ import print_function
import numpy as np
from sklearn import tree
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def creatDataSet():
    data=[]
    labels=[]
    with open("data.txt") as ifile:
        for line in ifile:
            tokens=line.strip().split(' ')
            data.append([float(tk) for tk in tokens[:-1]])
            labels.append(tokens[-1])
    #特征数据
    x=np.array(data)
    #label分类的标签数据
    labels=np.array(labels)
    #预估结果的标签数据
    y=np.zeros(labels.shape)
    #标签转换为0/1
    y[labels=='fat']=1
    #print(data, '-------', x, '-------', labels, '-------', y)
    return x, y

#利用信息熵作为划分标准，对决策树进行训练
def predict_train(x_train,y_train):
    clf=tree.DecisionTreeClassifier(criterion='entropy')
    clf.fit(x_train,y_train)
    '''系数反应每个特征的影响力，越大表示该特征在分类中起到的作用越大'''
    print('feature_importance_:%s' %clf.feature_importances_)
    '''测试结果的打印'''
    y_pre=clf.predict(x_train)
    #print(y_pre)
    #print(y_train)
    print(np.mean(y_pre==y_train))
    return y_pre,clf

#准确率与召回率
def show_precision_recall(x,y,clf,y_train,y_pre):
    precision, recall, thresholds = precision_recall_curve(y_train, y_pre)
    # 计算全量的预估结果
    answer = clf.predict_proba(x)[:, 1]
    '''
    precisicon 准确率
    recall 召回率
    f1-score 准确率和召回率的一个综合得分
    '''
    target_names=['thin','fat']
    #print(classification_report(y,answer,target_names=target_names))
    print(answer)
    #print(y)

#可视化输出，把决策树结构写入文件
def show_pdf(clf):
    import pydotplus
    from sklearn.externals.six import StringIO
    dot_data=StringIO()
    tree.export_graphviz(clf,out_file=dot_data)
    graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf('tree.pdf')

if __name__=='__main__':
    x,y=creatDataSet()
    '''拆分数据集'''
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
    print('拆分数据：',x_train,x_test,y_train,y_test)
    #得到训练的预测结果集
    y_pre,clf=predict_train(x_train,y_train)
    #展现准确率和zhaohuil
    show_precision_recall(x,y,clf,y_train,y_pre)
    #可视化输出
    #show_pdf(clf)
