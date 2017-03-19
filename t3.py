#!env python3
# --*-- coding:gbk --*--

# testing code of chapter 3

import trees
import treePlotter

(ds, labels) = trees.createDataSet()
print('data set :\n', ds)
print(trees.calcShannonEnt(ds))

print('.................')

print(trees.chooseBestFeatureToSplit(ds))

my_tree = trees.createTree(ds, labels)

my_tree =treePlotter.retrieveTree(1)

treePlotter.createPlot(my_tree)