#!env python3
# --*-- coding:gbk --*--

import svmMLiA


def test1():
    dataArr, labelArr = svmMLiA.loadDataSet('testSet.txt')
    return dataArr, labelArr


def test2(dataArr, labelArr):
    b, alphas = svmMLiA.smoSimple(dataArr, labelArr, 0.6, 0.001, 40)

    print(b)
    print('-------------')
    for i in range(len(alphas)):
        if alphas[i] > 0:
            print(dataArr[i], labelArr[i])


if __name__=='__main__':
    dataArr, labelArr = test1()
    test2(dataArr, labelArr)

