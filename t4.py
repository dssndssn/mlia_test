#!env python3
# --*-- coding:gbk --*--

# testing code of chapter 3

import bayes
from numpy import *

listOPosts, listClasses = bayes.loadDataSet()
myVocabList = bayes.createVocabList(listOPosts)

trainMat = []
for postinDoc in listOPosts:
    trainMat.append(bayes.setOfWords2Vec(myVocabList,postinDoc))

p0V, p1V, pAb = bayes.trainNB0(trainMatrix=trainMat, trainCategory=listClasses)

print(pAb, p0V, p1V)
