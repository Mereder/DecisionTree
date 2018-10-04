# 决策树的 SKlearn Decision 实现
import pandas as pd
from sklearn.externals.six import StringIO
from sklearn.metrics import  accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import tree

import pydotplus.graphviz

import os
os.environ["PATH"] += os.pathsep + 'D:/Graphviz/bin/'

if __name__ == '__main__':
    # 加载文件
    with open('page-blocks.data', 'r') as fr:  #加载文件
        page_blocks = [inst.strip().split() for inst in fr.readlines()] #处理文件
    pageblocks_target = []   #提取每组数据的类别，保存在列表里
    for each in page_blocks:
        pageblocks_target.append(each[-1])
    # print(pageblocks_target)
    pageblocks_Labels = ['height','lenght','area','eccen','p_block','p_and',
                         'mean_tr','blackpix','blackand','wb_trans']            # 特征标签

    pageblocks_List = []                                                 # list 临时列表
    pageblocks_Dict = {}                                                 #  字典 用于生成 pandas

    for each_Label in pageblocks_Labels:                                 #  生成字典
        for each in page_blocks:
            pageblocks_List.append(each[pageblocks_Labels.index(each_Label)]) # 读取每一列
        pageblocks_Dict[each_Label] = pageblocks_List
        pageblocks_List = []
    # print(pageblocks_Dict)
    pageblocks_pd = pd.DataFrame(pageblocks_Dict)                        # 生成pandas
    # print(pageblocks_pd)
    pb = LabelEncoder()                                                  # 创建Encoder对象 用于序列化！！！！！！！
    for col in pageblocks_pd.columns:                                    # 开始序列化
        pageblocks_pd[col] = pb.fit_transform(pageblocks_pd[col])

    clf = tree.DecisionTreeClassifier(criterion="entropy",max_depth=4)   # 创建DecisionTreeClassifier()类
    clf = clf.fit(pageblocks_pd.values.tolist(), pageblocks_target)      # 使用数据构建决策树

    dot_data = StringIO()

    tree.export_graphviz(clf, out_file=dot_data,  # 绘制决策树
                         feature_names=pageblocks_pd.keys(),
                         class_names=clf.classes_,
                         filled=True, rounded=True,
                         special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("tree.pdf")
    # 数据测试
    # x为数据集的feature 部分，y为label.
    x_train, x_test, y_train, y_test = train_test_split(page_blocks, pageblocks_target, test_size=0.2)

    Test_x = []
    for test in x_test:
        test = test[0:-1]
        Test_x.append(test)

    print(accuracy_score(clf.predict(Test_x),y_test))

