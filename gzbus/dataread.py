import psycopg2
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
import sklearn.metrics as metrics
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt

def creatDataSet():
    con=psycopg2.connect(host="127.0.0.1",user="postgres",password="lordpeng",database="postgres")
    cursor=con.cursor()
    cursor.execute("SELECT * FROM consu10")
    data=cursor.fetchall()
    columnDes=cursor.description
    columnNames=[columnDes[i][0] for i in range(len(columnDes))]
    df = pd.DataFrame([list(i) for i in data],columns=columnNames)
    countno=df["countno"]
    summoney=df["summoney"]
    deldate=df["delconsum"]
    chamoney=df["chargemoney"]
    leave=df.iloc[:,4]
    countno_div=pd.qcut(countno,5,labels=[1,2,3,4,5])
    summoney_div=pd.qcut(summoney,5,labels=[1,2,3,4,5])
    deldate_div=pd.cut(deldate,5,labels=[5,4,3,2,1])
    #chamoney_div=pd.qcut(chamoney,5,labels=[1,2,3,4,5])
    data=[]
    labels=[]
    for i in range(len(countno)):
        temp=[]
        temp.append(countno_div[i])
        temp.append(summoney_div[i])
        temp.append(deldate_div[i])


        # temp.append(countno[i]*countno[i])
        # temp.append(summoney[i]*summoney[i])
        #temp.append(deldate_div[i])
        data.append(temp)
        labels.append(leave[i])
    #特征数据
    x=np.array(data)
    #分类的标签数据
    labels=np.array(labels)
    return x,labels

#利用信息熵作为划分标准，对决策树进行训练
def predict_train(x_train,y_train,x_test,y_test):
    clf=tree.DecisionTreeClassifier(criterion='entropy',max_depth=8,min_samples_split=36)
    clf.fit(x_train,y_train)
    '''系数反应每个特征的影响力，越大表示该特征在分类中起到的作用越大'''
    print('feature_importance_:%s' % clf.feature_importances_)
    '''测试结果的打印'''
    train_est=clf.predict(x_train)
    train_est_p=clf.predict_proba(x_train)[:,1]
    test_est=clf.predict(x_test)
    test_est_p=clf.predict_proba(x_test)[:,1]
    print("训练集准确率")
    print(np.mean(y_train==train_est))
    print("测试集准确率")
    print(np.mean(y_test==test_est))
    print(metrics.confusion_matrix(y_test, test_est, labels=[0, 1]))  # 混淆矩阵
    print(metrics.classification_report(y_test, test_est))  # 计算评估指标
    # fpr_test,tpr_test,th_test=metrics.roc_curve(y_test,test_est_p)
    # fpr_train,tpr_train,th_train=metrics.roc_curve(x_test,train_est_p)
    # plt.figure(figsize=[6,6])
    # plt.plot(fpr_test,tpr_test,'b')
    # plt.plot(fpr_train,tpr_train,'r')
    # plt.show()
    return clf

def testbest(train_data,train_target):
    param_grid = {
        'criterion': ['entropy', 'gini'],
        'max_depth': [4, 5, 6, 7, 8, 9, 10],
        'min_samples_split': [12, 16, 20, 24, 28, 32, 36]
    }
    clf = tree.DecisionTreeClassifier()
    clfcv = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='roc_auc', cv=4)
    clfcv.fit(train_data, train_target)
    print(clfcv.best_params_)

def show_pdf(clf):
    import pydotplus
    from sklearn.externals.six import StringIO
    import os
    os.environ["PATH"] += os.pathsep + 'H:/graphviz/bin/'

    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf('tree.pdf')

if __name__=='__main__':
    x,y=creatDataSet()
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
    clf=predict_train(x_train,y_train,x_test,y_test)
    #testbest(x_train,y_train)
    show_pdf(clf)
