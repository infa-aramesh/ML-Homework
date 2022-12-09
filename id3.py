from hw0 import readAndProcess
import numpy as np
import matplotlib.pyplot as plt
import collections
import math
from treelib import Tree
from HeartDataSet import HeartDataSet
import sys
import statistics as stat
import random

dataX_train, dataY_train, exampleNames_train, dataX_tune, dataY_tune, exampleNames_tune, dataX_test, dataY_test, exampleNames_test = readAndProcess('id3')
# min = np.amin(dataX_train, axis = 0)
# print("Min: ", min)
# max = np.amax(dataX_train, axis = 0)
# print("Max: ", max)
# uniqueVar, counter = np.unique(dataY_train, return_counts=True)
# print(uniqueVar, counter)
def discretize(dataX_train):
    # 0, 3, 4, 7, 9
    # bins = {
    #     0: 12,
    #     3: 12,
    #     4: 24,
    #     7: 22,
    #     9: 12
    # }
    # bins = {
    #     0: 41,
    #     3: 48,
    #     4: 133,
    #     7: 86,
    #     9: 39
    # }
    bins = {
        0: 4,
        3: 5,
        4: 7,
        7: 5,
        9: 5
    }
    _, num_cols = dataX_train.shape
    splits = [[] for _ in range(num_cols)]
    for i in range(dataX_train.shape[1]):
        uniqueValues = np.unique(dataX_train[:, i])
        # print(i, len(uniqueValues))
        categorySize = math.floor(len(uniqueValues) / bins.get(i, -1))
        if len(uniqueValues) > categorySize and categorySize > 0:
            keys = sorted(uniqueValues)
            splitLen = len(keys) // categorySize
            for k in range(0, categorySize):
                splits[i].append(keys[k * splitLen])
            for val in range(dataX_train.shape[0]):
                for num in reversed(range(categorySize)):
                    if dataX_train[val][i] >= splits[i][num]:
                        dataX_train[val][i] = num
                        break
    return splits

def discretizeTuneOrTest(data, splits):
    for col in range(data.shape[1]):
        if splits[col]: 
            for row in range(data.shape[0]):
                for num in reversed(range(len(splits[col]))):
                    if data[row][col] >= splits[col][num]:
                        data[row][col] = num
                        break
        

def findEntropy(dataX_train, column, Y):
    # for each unique category in column find conditional probability of result 
    entropy = 0
    # counts = collections.Counter(dataX_train[:, column])
    unique, count = np.unique(dataX_train[:, column], return_counts=True)
    for ite in range(len(unique)):
        dictOfResultCategory = dict.fromkeys(np.unique(Y), 0)
        for i in range(dataX_train.shape[0]):
            if dataX_train.ndim == 1:
                dataX_train = np.array([dataX_train])
            if len(Y) == 1:
                Y = np.array([Y])
            if dataX_train[i, column] == unique[ite]:
                dictOfResultCategory[Y[i][0]] = dictOfResultCategory[Y[i][0]] + 1
        # for each result category find condional probability
        condProb = 0
        for key in dictOfResultCategory.keys():
            if dictOfResultCategory[key] == 0 or count[ite] == 0:
                continue
            condProb = - ((dictOfResultCategory[key]/count[ite]) * math.log((dictOfResultCategory[key]/count[ite]), 2)) + condProb
        probabiltyOfItem = count[ite] / dataX_train[:, column].shape[0]
        itemProb = probabiltyOfItem * condProb
        entropy = entropy + itemProb
    return entropy

def findMinEntropy(dataX_train, dataY_train, selectedColumn):
    entropy = []
    if dataX_train.ndim == 1:
        dataX_train = np.array([dataX_train])
    for i in range(dataX_train.shape[1]):
        if not selectedColumn.__contains__(i):
            entropy.append(findEntropy(dataX_train, i, dataY_train))
        else:
            entropy.append(sys.maxsize)
    minEntropy = np.argmin(entropy)
    # print(entropy)
    return minEntropy, dataX_train

def traverseDescisionTree(data_TuneOrTest, Y, tree):
    positive, negative = 0, 0
    res = []
    node = tree.get_node(tree.root)
    splits = node.data.splits
    discretizeTuneOrTest(data_TuneOrTest, splits)
    for row in range(data_TuneOrTest.shape[0]):
        #traverse the decision tree for each row
        input = data_TuneOrTest[row]
        node = tree.get_node(tree.root)
        matchingChild = True
        while not node.is_leaf() and matchingChild:
            matchingChild = False
            children = tree.children(node.identifier)
            for child in children:
                if child.data.result is None:
                    col = child.data.attributeName 
                    if input[col] == child.data.value:
                        node = child
                        matchingChild = True
                        break
                else:
                    node = child
                    matchingChild = True
                    break
        if node.data.result is None:
            result = stat.mode(node.data.y.flatten())
        else:
            result = node.data.result
        if result == Y[row][0]:
            positive = positive + 1
        else:
            negative = negative + 1
        res.append(result)
    return positive, negative, res

def createDecisionTree(dataX_train, dataY_train, exampleNames_train, selectedColumn, algo, d):
    tree = Tree()
    splits = discretize(dataX_train)
    root = tree.create_node("root", "root", data=HeartDataSet("", None, dataX_train, dataY_train, exampleNames_train, splits))
    depth = 0
    q = []
    q.append(root)
    rootId = 0
    while len(q) > 0 and depth < d:
        size = len(q)
        for i in range(size):
            node = q.pop(0)
            rootTag = node.tag
            if algo == 'id3':
                bestColumn, dataX_train = findMinEntropy(node.data.x, node.data.y, selectedColumn)
            elif algo == 'rf':
                bestColumn, dataX_train = random.randint(0, dataX_train.shape[1] - 1), node.data.x
                if dataX_train.ndim == 1:
                    dataX_train = np.array([dataX_train])
            # bestColumn, dataX_train = findMinEntropy(node.data.x, node.data.y, selectedColumn)
            splits = node.data.splits
            # for each value in bestColumn of root node create a node then traverse each node 
            uniqueValues = np.unique(dataX_train[:,bestColumn])
            valuesDict = {}
            for val in uniqueValues:
                valuesDict[val] = HeartDataSet("", None)
            for i in range(dataX_train.shape[0]):
                if valuesDict[dataX_train[i, bestColumn]].x is None:
                    valuesDict[dataX_train[i, bestColumn]].x = np.array(dataX_train[i])
                    valuesDict[dataX_train[i, bestColumn]].y = np.array(dataY_train[i])
                    valuesDict[dataX_train[i, bestColumn]].exampleName = np.array(exampleNames_train[i])
                else:
                    valuesDict[dataX_train[i, bestColumn]].x = np.vstack([valuesDict[dataX_train[i, bestColumn]].x, dataX_train[i]])
                    valuesDict[dataX_train[i, bestColumn]].y = np.vstack([valuesDict[dataX_train[i, bestColumn]].y, dataY_train[i]])
                    valuesDict[dataX_train[i, bestColumn]].exampleName = np.vstack([valuesDict[dataX_train[i, bestColumn]].exampleName, exampleNames_train[i]])
            for val in uniqueValues:
                q.append(tree.create_node(rootId, rootId, data=HeartDataSet(bestColumn, val, valuesDict[val].x, valuesDict[val].y, valuesDict[val].exampleName, splits), parent=rootTag))
                rootId = rootId + 1
        depth = depth + 1
        selectedColumn.add(bestColumn)
        # tree.show(data_property=["attributeName", "value"])
    while len(q) > 0:
        node = q.pop(0)
        rootTag = node.tag
        mode = stat.mode(node.data.y.flatten())
        tree.create_node(rootId, rootId, data=HeartDataSet(None, None, result=mode), parent=rootTag)
        isLeaf = tree.get_node(rootId).is_leaf()
        rootId = rootId + 1
    # tree.show(data_property=["attributeName", "value"])
    # tree.show(data_property="attributeName")
    # tree.show(data_property="value")
    # tree.show(data_property="result")
    return tree
    # create result node as leaf
    # traverse the tree with tune data and calculate accuracy

def evaluateDecisionTree(dataX_train, dataY_train, exampleNames_train, dataX_tune, dataY_tune, exampleNames_tune, dataX_test, dataY_test, exampleNames_test, algo, depth):
    folds = 9
    cumulativePostive, cumulativeNegative = 0, 0
    results = []
    bestRate, bestTree = 0, -1
    for i in range(folds):
        selectedColumn = set()
        dataX_train_bkp = np.copy(dataX_train)
        dataX_tune_bkp = np.copy(dataX_tune)
        dataX_test_bkp = np.copy(dataX_test)
        sampleSize = dataX_test_bkp.shape[0]
        tree = createDecisionTree(dataX_train, dataY_train, exampleNames_train, selectedColumn, algo, depth)
        results.append(tree)
        postive, negative, res = traverseDescisionTree(dataX_tune, dataY_tune, tree)
        cumulativePostive = cumulativePostive + postive
        cumulativeNegative = cumulativeNegative + negative
        succesRate = (postive / (postive + negative)) * 100
        failureRate = (negative / (postive + negative)) * 100 
        if succesRate > bestRate:
            bestRate = succesRate
            bestTree = i
        print(f'----------SuccessRate in Tune {i}----------', succesRate)
        print(f'----------FailureRate in Tune {i}----------', failureRate)
        # traverseDescisionTree(dataX_test, dataY_test, tree)
        dataX_train = dataX_train_bkp
        dataX_tune = dataX_tune_bkp
        dataX_test = dataX_test_bkp
        dataX_train = np.concatenate((dataX_train, dataX_tune), axis=0)
        dataY_train = np.concatenate((dataY_train, dataY_tune), axis=0)
        exampleNames_train = np.concatenate((exampleNames_train, exampleNames_tune), axis=0)
        dataX_tune = dataX_train[0:sampleSize, :]
        dataY_tune = dataY_train[0:sampleSize, :]
        exampleNames_tune = exampleNames_tune[0:sampleSize, :]
        dataX_train = dataX_train[sampleSize + 1:, :]
        dataY_train = dataY_train[sampleSize + 1:, :]
        exampleNames_train = exampleNames_train[sampleSize + 1:, :]
    cumulativePostivePercent = (cumulativePostive / (cumulativePostive + cumulativeNegative)) * 100
    cumulativeNegativePercent = (cumulativeNegative / (cumulativePostive + cumulativeNegative)) * 100
    print("----------Cumulative SuccessRate in Tune----------", cumulativePostivePercent)
    print("----------Cumulative FailureRate in Tune----------", cumulativeNegativePercent)
    if algo == 'rf':
        return cumulativePostivePercent
    # return results[bestTree]
    return results, bestTree

def evaluateTest(trees, bestTree, dataX_test, dataY_test, exampleNames_test):
    # for tree in trees:
    positive, negative, result = traverseDescisionTree(dataX_test, dataY_test, trees[bestTree])
    succesRate = (positive / (positive + negative)) * 100
    failureRate = (negative / (positive + negative)) * 100
    print("----------SuccessRate in Test----------", succesRate)
    print("----------FailureRate in Test----------", failureRate)

# trees, bestTree = evaluateDecisionTree(dataX_train, dataY_train, exampleNames_train, dataX_tune, dataY_tune, exampleNames_tune, dataX_test, dataY_test, exampleNames_test, 'id3', 3)
# evaluateTest(trees, bestTree, dataX_test, dataY_test, exampleNames_test)
