import adaboost
from numpy import *


datMat, class_labels = adaboost.loadSimpData()

# print(datMat)
# print(class_labels)

# d = mat(ones((5, 1))/5)
# print(adaboost.buildStump(datMat, class_labels, d))

classifierArray = adaboost.adaBoostTrainDS(datMat, class_labels, 30)

print(adaboost.adaClassify([[5, 5],[0,0]], classifierArray))