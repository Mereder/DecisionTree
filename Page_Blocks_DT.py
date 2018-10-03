import operator

from math import log


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

def majorityCnt(classList):
    classCount = { }
    for vote in classList:
        if  vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]      # 返回出现次数最多的元素


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


def splitDataSet(dataSet, axis, value):
    returnDataset = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]                     # 去掉axis 的特征 :axis 含义 从0到axis-1 共 axis个元素
            reducedFeatVec.extend(featVec[axis + 1:])
            returnDataset.append(reducedFeatVec)                # 将符合条件的添加到返回的数据集
    return returnDataset


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

def createTree(dataSet, Labels):
    classList = [example[-1] for example in dataSet]             # 取分类标签 1,2,3,4,5
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:                                     # 遍历完所有特征时返回出现次数最多的类标签
        return majorityCnt(classList)
    bestFeature = chooseBestFeatureToSplit(dataSet)              # 选择最优特征的下标 是 bestFeature
    # print(bestFeature)
    bestFeatureLabel = Labels[bestFeature]                       # 最优特征的标签
    # featLabels.append(bestFeatureLabel)                          # 汇入 最优特征标签列表
    myTree = {bestFeatureLabel: {}}                              # 根据最优特征的标签生成树
    del (Labels[bestFeature])                                    # 删除已经使用过的特征标签
    featValues = [example[bestFeature] for example in dataSet]   # 得到训练集中所有最优特征的属性值
    uniqueVals = set(featValues)                                 # 去掉重复的属性值
    for value in uniqueVals:                                     # 遍历特征，创建决策树。
        subLabels = Labels[:]
        myTree[bestFeatureLabel][value] = createTree(splitDataSet(dataSet, bestFeature, value), subLabels)
    return myTree


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
    featLabels = list(Labels);
    myTree = createTree(dataSet,Labels)

    # testVec = ['5', '7', '35', '1.400', '.400','.657',  '2.33',  '14',  '23',  '6']  # 测试数据
    # 7   1      7   .143 1.00 1.00     7.00      7      7      1 4
    #    8 231   1848 28.875 .872 .882   179.00   1611   1630      9
    # 158 167  26386  1.057 .160 .276     8.24   4213   7279    511 5
    numTestVecs = 300
    errorCount = 0.0

    for i in range(numTestVecs):
        # 暂时取50个作为测试集
        result = classify(myTree, featLabels,dataSet[i])
        print("分类类别:%s\t真实类别:%s" % (result, dataSet[i][-1]))
        if result != dataSet[i][-1]:
            errorCount += 1.0
    print("错误率:%f%%" %(errorCount/float(numTestVecs)*100))
    # testVec = ['158', '167', '26386', '1.057', '0.160', '.276',  '8.24', '4213', '7279', '511']

    # result = classify(myTree, featLabels, testVec)
    # print(result)


    