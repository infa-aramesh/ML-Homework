from tempfile import TemporaryFile
import numpy as np
from dataReader import collectTrainTuneTestSets
from preprocess import preproc

def readAndProcess(algo):
    fold = 10
    list=[]
    with open("processed.cleveland.data", "r") as file:
        list = [float(element) for line in file for element in line.strip().split(',')]
    np.set_printoptions(suppress=True)
    grid = np.array(list)
    features = np.array(open("attributes.data", "r").read().split(","))
    dataXY = grid.reshape(297, 14)
    exampleNames = np.array([['orgEx' + str(i)] for i in range(dataXY.shape[0])], dtype=(np.unicode_, 16))
    dtypeX = np.dtype([("age", np.float64), ("sex", np.float64), ("cp", np.float64), ("trestbps", np.float64), ("chol", np.float64), ("fbs", np.float64), 
    ("restecg", np.float64), ("thalach", np.float64), ("exang", np.float64), ("oldpeak", np.float64), ("slope", np.float64), ("ca", np.float64),
    ("thal", np.float64)])
    dtypeY =  np.dtype([("y", np.float64)])
    dtypeXY = dtypeX.descr + dtypeY.descr
    dataXY.astype(dtypeXY)
    # print(dataXY)
    if(algo == 'kNN'):
        dataX, mean, std = preproc(dataXY[:,:13])
    else:
        dataX = dataXY[:,:13]
    
    dataY = dataXY[:,[13]]
    # use mean and std for future queries

    # print("Mean Before: ", None if algo != 'kNN' else mean)
    # print("SD Before: ", None if algo != 'kNN' else std)
    # print("Mean After: ", np.mean(dataX, axis = 0))
    # print("SD After: ", np.std(dataX, axis = 0))
    # print(dataXY)
    files = []
    outfile = TemporaryFile()
    for i in range(fold):
        np.savez(outfile,dataX[i:dataX.shape[0]:fold], dataY[i:dataY.shape[0]:fold], exampleNames[i:exampleNames.shape[0]:fold])
        _=outfile.seek(0)
        files.append(np.load(outfile))
    # print(dataX)
    if(algo == 'rf'):
        return files
    return collectTrainTuneTestSets(files, 2, 8)

    # print(exampleNames_tune)
    # print(dataY)
    # print(features)
    # print(exampleNames)

readAndProcess("kNN")