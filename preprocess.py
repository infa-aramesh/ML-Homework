# from hw0 import readAndProcess
import numpy as np

# dataX_train, dataY_train, exampleNames_train, dataX_tune, dataY_tune, exampleNames_tune, dataX_test, dataY_test, exampleNames_test = readAndProcess()

def preproc(array):
    mean = np.mean(array, axis = 0)
    
    for i in range(array.shape[1]):
        array[:, i] = (array[:, i] - mean[i])
    sd = np.std(array, axis = 0)
    for i in range(array.shape[1]):
            array[:, i] = (array[:, i] / sd[i])
    return array, mean, sd

# dataX_train = preprocess(dataX_train)
# dataX_tune = preprocess(dataX_tune)
# dataX_test = preprocess(dataX_test)
# print(dataX_train)
# print("Tune")
# print(dataX_tune)