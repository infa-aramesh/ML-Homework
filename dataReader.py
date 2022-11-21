import numpy as np
def collectTrainTuneTestSets(files, tuneFold, testFold = None):
    if testFold is None:
        testFold = (tuneFold + 1) % 10
    train_sample = 0
    if (tuneFold - 1) % 10 >= 0:
        train_sample = (tuneFold - 1) % 10 
    else:
        train_sample = (tuneFold + 2) % 10
    if tuneFold < len(files) and ((tuneFold > 0 and (tuneFold + 1) % 10 < len(files)) or (tuneFold + 2) < len(files)):
        dataX_tune = files[tuneFold][files[tuneFold].files[0]]
        dataY_tune = files[tuneFold][files[tuneFold].files[1]]
        exampleNames_tune = files[tuneFold][files[tuneFold].files[2]]
        dataX_test = files[testFold][files[testFold].files[0]]
        dataY_test = files[testFold][files[testFold].files[1]]
        exampleNames_test = files[testFold][files[testFold].files[2]]
        dataX_train = files[train_sample][files[train_sample].files[0]]
        dataY_train = files[train_sample][files[train_sample].files[1]]
        exampleNames_train = files[train_sample][files[train_sample].files[2]]
        for i in range(0, len(files)):
            if(i != testFold and i != tuneFold and i != train_sample):
                X = files[i][files[i].files[0]]
                Y = files[i][files[i].files[1]]
                names = files[i][files[i].files[2]]
                dataX_train = np.concatenate((dataX_train,X))
                dataY_train = np.concatenate((dataY_train, Y))
                exampleNames_train = np.concatenate((exampleNames_train, names))
        return dataX_train, dataY_train, exampleNames_train, dataX_tune, dataY_tune, exampleNames_tune, dataX_test, dataY_test, exampleNames_test
    else:
        raise Exception(f'TuneFold value is invalid: {tuneFold} and file size: {len(files)}')
