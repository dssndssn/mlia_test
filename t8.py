#############################
# test script for Charpter 8
#############################

import regression
from numpy import *
import matplotlib.pyplot as plt

x_arr, y_arr = regression.loadDataSet('ex0.txt')

print(x_arr,y_arr)

ws = regression.standRegres(x_arr, y_arr)

print(ws)

x_mat = mat(x_arr)
y_mat = mat(y_arr)

y_hat = x_mat * ws


# chapter 8.1

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x_mat[:,1].flatten().A[0], y_mat.T[:,0].flatten().A[0])
x_copy = x_mat.copy()
x_copy.sort(0)
y_hat = x_copy * ws
ax.plot(x_copy[:,1],y_hat)
plt.show()

y_hat = x_mat * ws

print(corrcoef(y_hat.T, y_mat))

# chapter 8.2

x_arr, y_arr = regression.loadDataSet('ex0.txt')

print(regression.lwlr(x_arr[0], x_arr, y_arr, 1.0))
y_hat = regression.lwlrTest(x_arr, x_arr, y_arr, 0.01)

x_mat = mat(x_arr)
srt_ind = x_mat[:, 1].argsort(0)
x_sort = x_mat[srt_ind][:, 0, :]

fig =plt.figure()
ax = fig.add_subplot(111)
ax.plot(x_sort[:, 1], y_hat[srt_ind])

ax.scatter(x_mat[:,1].flatten().A[0], mat(y_arr).T[:,0].flatten().A[0], s=2, c='red')
plt.show()