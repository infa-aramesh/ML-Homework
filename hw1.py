from hw0 import readAndProcess
import numpy as np
import math as  m
import statistics as stat

def kNN(k, dataX_train, dataY_train, exampleNames_train, dataX_tune_or_test, dataY_tune_or_test, cumulativeScore):
    # print(dataX_train)
    # print(exampleNames_train)
    # dtypeX = np.dtype([("age", np.float64), ("sex", np.float64), ("cp", np.float64), ("trestbps", np.float64), ("chol", np.float64), ("fbs", np.float64), 
    # ("restecg", np.float64), ("thalach", np.float64), ("exang", np.float64), ("oldpeak", np.float64), ("slope", np.float64), ("ca", np.float64),
    # ("thal", np.float64)])
    # dtypeEx = np.dtype([("exampleName", np.unicode_, 16)])
    # dtypeY =  np.dtype([("y", np.float64)])
    # dtypeDis = np.dtype([("distance", np.float64)])
    # npDtype = dtypeEx.descr + dtypeX.descr + dtypeY.descr + dtypeDis.descr
    # predict values
    score = dict.fromkeys(k, 0)
    negativeScore = dict.fromkeys(k, 0)
    for sample in range(dataX_tune_or_test.shape[0]):
        distanceFromTrain = []
        for train in dataX_train:
            distanceFromTrain.append([np.linalg.norm(dataX_tune_or_test[sample] - train)])
        data = np.concatenate((np.array(exampleNames_train, dtype='O'), dataX_train), axis = 1)
        data = np.concatenate((data, dataY_train), axis = 1)
        data = np.concatenate((data, distanceFromTrain), axis = 1)
        data = data[data[:, data.shape[1] - 1].argsort()]
        for neighbor in k:
            nearestK = data[:neighbor,data.shape[1] - 2]
            mode = stat.mode(nearestK)
            if(dataY_tune_or_test[sample] == mode):
                score[neighbor] = score[neighbor] + 1
                cumulativeScore[neighbor] = cumulativeScore[neighbor] + 1
            else:
                negativeScore[neighbor] = negativeScore[neighbor] + 1
    return score, negativeScore

def kNNWithFolds():
    dataX_train, dataY_train, exampleNames_train, dataX_tune, dataY_tune, exampleNames_tune, dataX_test, dataY_test, exampleNames_test = readAndProcess('kNN')
    dataX_train = np.concatenate((dataX_train, dataX_tune))
    dataY_train = np.concatenate((dataY_train, dataY_tune))
    exampleNames_train = np.concatenate((exampleNames_train, exampleNames_tune))
    k = [1, 3, 7, 10, 15, 25, 51, 101]
    examplesInFold = dataX_tune.shape[0]
    cumulativeScore = dict.fromkeys(k, 0)
    cumulativeNegativeScore = dict.fromkeys(k, 0)
    for i in range(m.floor(dataX_train.shape[0] / examplesInFold)):
        dataX_tune = dataX_train[0:examplesInFold, :]
        dataX_train = dataX_train[examplesInFold + 1:, :]
        dataY_tune = dataY_train[0:examplesInFold, :]
        dataY_train = dataY_train[examplesInFold + 1:, :]
        exampleNames_tune = exampleNames_train[0:examplesInFold, :]
        exampleNames_train = exampleNames_train[examplesInFold + 1:, :]
        score, negativeScore = kNN(k, dataX_train, dataY_train, exampleNames_train, dataX_tune, dataY_tune, cumulativeScore)
        for key in score.keys():
            cumulativeScore[key] = cumulativeScore[key] + score[key]
        for key in score.keys():
            cumulativeNegativeScore[key] = cumulativeNegativeScore[key] + negativeScore[key]
        dataX_train = np.concatenate((dataX_train, dataX_tune))
        dataY_train = np.concatenate((dataY_train, dataY_tune))
        exampleNames_train = np.concatenate((exampleNames_train, exampleNames_tune))
    print(cumulativeScore)
    print(cumulativeNegativeScore)
    positivePercentForK = dict()
    for key in cumulativeScore:
        positivePercentForK[key] = (cumulativeScore[key] / (cumulativeScore[key] + cumulativeNegativeScore[key])) * 100
    print(positivePercentForK)
kNNWithFolds()
def kNNTestFold():
    bestK = 10
    k = [bestK]
    cumulativeScore = dict.fromkeys(k, 0)
    dataX_train, dataY_train, exampleNames_train, dataX_tune, dataY_tune, exampleNames_tune, dataX_test, dataY_test, exampleNames_test = readAndProcess('kNN')
    score, negativeScore = kNN(k, dataX_train, dataY_train, exampleNames_train, dataX_test, dataY_test, cumulativeScore)
    positivePercent = (score[bestK] / (score[bestK] + negativeScore[bestK])) * 100
    negativePercent = (negativeScore[bestK] / (score[bestK] + negativeScore[bestK])) * 100
    print(positivePercent)
    print(negativePercent)
# kNNTestFold()
    
