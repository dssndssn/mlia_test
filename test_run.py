#!env python3
# --*-- coding:gbk --*--

# 测试下

import kNN
from numpy import *

#test classify 0
(group, labels) = kNN.createDataSet()
print(group)
print(labels)
print(kNN.classify0([0.3, 0.2], group, labels, 3))

#test loading data
dating_mat,dating_labels = kNN.file2matrix('datingTestSet2.txt')
norm_dating_mat, ranges, min_vals = kNN.autoNorm(dating_mat)

print(norm_dating_mat)
print(ranges)
print(min_vals)

# PILOT
# import matplotlib
# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(norm_dating_mat[:, 0],norm_dating_mat[:, 1], 15*array(dating_labels), 15*array(dating_labels))
# plt.show()

kNN.handwritingClassTest()
