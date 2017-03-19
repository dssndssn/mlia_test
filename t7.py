import adaboost
from numpy import *


datMat, class_labels = adaboost.loadSimpData()

# print(datMat)
# print(class_labels)

# d = mat(ones((5, 1))/5)
# print(adaboost.buildStump(datMat, class_labels, d))

classifierArray = adaboost.adaBoostTrainDS(datMat, class_labels, 30)

print(classifierArray)
print(adaboost.adaClassify([[5, 5],[0,0]], classifierArray))

dat_arr, label_arr = adaboost.loadDataSet('horseColicTraining2.txt')
classifierArray = adaboost.adaBoostTrainDS(dat_arr, label_arr, 10)

test_arr, test_label_arr = adaboost.loadDataSet('horseColicTest2.txt')

prediction10 = adaboost.adaClassify(test_arr, classifierArray)

err_arr = mat(ones((67,1)))
print(err_arr[prediction10!=mat(test_label_arr).T].sum())

