import operator

from math import log

import matplotlib.pyplot as plt


decisionNode = dict(boxstyle="sawtooth",fc="0.8")
leafNode = dict(boxstyle = "round4",fc = "0.8")
arrow_args = dict(arrowstyle="<-")

def plotNode(nodeText, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeText,xy=parentPt,xycoords = 'axes fraction',xytext=centerPt, textcoords='axes fraction',
		va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)

def createPlot(myTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[],yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(myTree))
    plotTree.totalD = float(getTreeDepth(myTree))
    plotTree.xOff = -0.5 / plotTree.totalW;plotTree.yOff = 1.0;
    plotTree(myTree, (0.5, 1.0), '')
    plt.show()

def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict:
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth

def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict:
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:numLeafs += 1
    return numLeafs

def retrieveTree(i):
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]

def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

def plotTree(myTree, parentPt, nodeTxt):#if the first key tells you what feat was split on
    numLeafs = getNumLeafs(myTree)  #this determines the x width of this tree
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]     #the text label for this node should be this
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
            plotTree(secondDict[key],cntrPt,str(key))        #recursion
        else:   #it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
#if you do get a dictonary you know it's a tree, and the first element will be another dict
























# 创建数据集
def creatDataSet():
    """
    :rtype: dataSet,Label
    """
    # 加载文件
    with open('page-blocks.data', 'r') as fr:  # 加载文件
        page_blocks = [inst.strip().split() for inst in fr.readlines()]  # 处理文件 最后一列的特征值未动
    # lenses_target = []  # 提取每组数据的类别，保存在列表里
    # for each in page_blocks:
    #    lenses_target.append(each[-1])
    # print(lenses_target)lenses_target
    # 特征标签：共有10项
    # height:  lenght:  area:      eccen:    p_black:
    # p_and:   mean_tr: blackpix   blackand  wb_trans
    Labels = ['height','lenght','area','eccen','p_block','p_and','mean_tr','blackpix','blackand','wb_trans']
    return page_blocks,Labels

# 投票选出分类标签最多的作为类标签
def majorityCnt(classList):
    classCount = { }
    for vote in classList:
        if  vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]      # 返回出现次数最多的元素

# 计算香浓熵
def calcShannonEnt(dataSet):
    numOfEntires = len(dataSet)                                 # 返回数据集的行数
    labelCount = { }                                            # 保存每个标签(Label)出现次数的字典
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCount.keys():
            labelCount[currentLabel] = 0
        labelCount[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCount:
        prob = float(labelCount[key]) / numOfEntires            # 选择该标签(Label)的概率
        shannonEnt -= prob * log(prob, 2)                       # 公式计算香浓熵
    return shannonEnt

# 划分数据集
def splitDataSet(dataSet, axis, value):
    returnDataset = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]                     # 去掉axis 的特征 :axis 含义 从0到axis-1 共 axis个元素
            reducedFeatVec.extend(featVec[axis + 1:])
            returnDataset.append(reducedFeatVec)                # 将符合条件的添加到返回的数据集
    return returnDataset

# 选择最优特征标签，对最优标签进行划分
def chooseBestFeatureToSplit (dataSet):
    numOfFeat = len(dataSet[0]) - 1                              # 特征值的数量 = 10
    baseEntropy = calcShannonEnt(dataSet)                        # 计算数据集的香浓熵
    print(baseEntropy)
    bestInforGain = 0.0                                          # 信息增益
    bestFeature = -1                                              # 最优信息的索引值 后边会更新
    # 获取dataSet 的每一个特征
    for i in range(numOfFeat): # i 取值0 1 2 3 4 5 6 7 8 9
        featList = [example[i] for example in dataSet]
        uniqueValue = set(featList)                              # 创建set集合{},元素不可重复
        newEntropy = 0.0                                         # 经验条件熵
        for value in uniqueValue:
            subDataSet = splitDataSet(dataSet, i, value)         # 对dataSet划分后的子集
            prob = len(subDataSet) / float(len(dataSet))         # 计算子集的概率  这个子集 是 包含 value 的子集 然后将value 去除掉了
            newEntropy += prob * calcShannonEnt(subDataSet)      # 根据公式计算经验条件熵
        infoGain = baseEntropy - newEntropy                      # 计算信息增益
        # print("第%d个特征的增益为%.3f" % (i, infoGain))			#打印每个特征的信息增益
        if(infoGain > bestInforGain):
            bestInforGain = infoGain                             # 更新信息增益，找到最大的信息增益
            bestFeature = i
    return bestFeature                                            # 返回信息增益最大的特征的索引值

# 递归建树
def createTree(dataSet, Labels):
    classList = [example[-1] for example in dataSet]             # 取分类标签 1,2,3,4,5
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:                                     # 遍历完所有特征时返回出现次数最多的类标签
        return majorityCnt(classList)
    bestFeature = chooseBestFeatureToSplit(dataSet)              # 选择最优特征的下标 是 bestFeature
    # print(bestFeature)
    bestFeatureLabel = Labels[bestFeature]                       # 最优特征的标签
    # featLabels.append(bestFeatureLabel)                        # 汇入 最优特征标签列表
    myTree = {bestFeatureLabel: {}}                              # 根据最优特征的标签生成树
    del (Labels[bestFeature])                                    # 删除已经使用过的特征标签
    featValues = [example[bestFeature] for example in dataSet]   # 得到训练集中所有最优特征的属性值
    uniqueVals = set(featValues)                                 # 去掉重复的属性值
    for value in uniqueVals:                                     # 遍历特征，创建决策树。
        subLabels = Labels[:]
        myTree[bestFeatureLabel][value] = createTree(splitDataSet(dataSet, bestFeature, value), subLabels)
    return myTree

# 根据所建的决策树进行分类处理测试集
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]                            # 获取下一个字典
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():                               # !!!!!!!!!!!!!! key 是 str的  testVec 是 float的
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key],featLabels,testVec)
            else: classLabel = secondDict[key]
    return classLabel


if __name__ == '__main__':
    dataSet,Labels = creatDataSet()
    numTestVecs = 300  # 测试数据数量 前300个
    errorCount = 0.0
    featLabels = list(Labels);
    # myTree = createTree(dataSet,Labels)
    myTree = createTree(dataSet,Labels)

    createPlot(myTree)
    # testVec = ['5', '7', '35', '1.400', '.400','.657',  '2.33',  '14',  '23',  '6']  # 测试数据
    # testVec = ['158', '167', '26386', '1.057', '0.160', '.276',  '8.24', '4213', '7279', '511']

    for i in range(numTestVecs):
        # 暂时取50个作为测试集
        result = classify(myTree, featLabels,dataSet[i])
        print("分类类别:%s\t真实类别:%s" % (result, dataSet[i][-1]))
        if result != dataSet[i][-1]:
            errorCount += 1.0
    print("错误率:%f%%" %(errorCount/float(numTestVecs)*100))

    # result = classify(myTree, featLabels, testVec)
    # print(result)
