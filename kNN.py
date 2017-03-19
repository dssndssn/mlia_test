#!env python3
# --*-- coding:gbk --*--

# 测试下

from numpy import *
import operator
from os import listdir

# python3 移植修改
# 1、iteritems 取消，改为items
# 2、

#sum/argsort只能对数值类型的数据源有效
def classify0(inX, dataSet, labels, k):
    '''
    :param inX: a sample of data, a python list
    :param dataSet: given dataset, numpy narray
    :param labels: a list of data labels, one label for each data sample in dataset
    :param k: number of voters to classify
    :return: label of inX
    '''
    assert (len(labels) == dataSet.shape[0])
    assert(len(inX)<=dataSet.shape[1])
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

#Create sampe dataset
def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

#read file to dataset
def file2matrix(filename):
    '''
    :param filename: text file of data
    :return: (dataset,label) dataset - a numpy narray; label - a python list
    '''

    file_lines = open(filename).readlines()
    return_mat = zeros((len(file_lines), 3))  # prepare matrix to return, only first three items is accpetable
    class_labels = []  # prepare labels return
    index = 0
    for line in file_lines:
        list_from_line = line.strip().split('\t')
        return_mat[index, :] = list_from_line[0:3]
        class_labels.append(int(list_from_line[-1]))
        index += 1
    return return_mat, class_labels


def autoNorm(dataSet):
    '''
    归一化特征值
    :param dataSet: 输入数据集
    :return:
        normDataSet 归一化的数据集
        ranges 分布范围list（最大值-最小值)
        minVals 最小值list
    '''
    minVals = dataSet.min(0)  # axis 0, min values
    maxVals = dataSet.max(0)  # axis 0, max values
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))  #init
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))  # element wise divide
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.05  # hold out 10% # 测试样本比例。其余的均用于训练
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')  # load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    mat_count = normMat.shape[0]
    numTestVecs = int(mat_count * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:mat_count, :], datingLabels[numTestVecs:mat_count], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print( "the total error rate is: %f" % (errorCount / float(numTestVecs)))
    print("%d errors among %d tests with %d trains." % (errorCount, numTestVecs, mat_count - numTestVecs))


def img2vector(filename):
    '''
    将手写字体文件，转换为一维数组
    :param filename: 文件名
    :return:
    '''
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('digits/trainingDigits')  # load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('digits/trainingDigits/%s' % fileNameStr)
    testFileList = listdir('digits/testDigits')  # iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('digits/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount / float(mTest)))