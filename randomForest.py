import numpy as np
import random
from hw0 import readAndProcess
from id3 import createDecisionTree
from id3 import traverseDescisionTree
from statistics import mean

files = readAndProcess('rf')
featureCount = [1, 2, 3, 4, 5, 7]
accuracy = dict.fromkeys(featureCount, [])
bootstraps = [[] for _ in range(len(files))]
testFold = 5
for i, file in enumerate(files):
    dataXY = np.concatenate((file[file.files[0]], file[file.files[1]]), axis = 1)
    dataXY = np.concatenate((dataXY, file[file.files[2]]), axis = 1, dtype='O')
    bootstrap_copies = []
    for _ in range(101):
        bootstrap_rows = np.random.choice(dataXY.shape[0], size=dataXY.shape[0], replace=True)
        bootstrap_copies = np.copy(dataXY[bootstrap_rows,:])
        bootstraps[i].append(bootstrap_copies)

def generateTreesForBootstrap(i, tuneFoldOrTestFold, dataX_tuneOrTest, dataY_tuneOrTest, exampleNames_tuneOrTest):
    dataX_train, dataY_train, exampleNames_train = None, None, None
    for j in range(10):
        if j == tuneFoldOrTestFold:
            dataX_tuneOrTest = files[tuneFoldOrTestFold][files[tuneFoldOrTestFold].files[0]]
            dataY_tuneOrTest = files[tuneFoldOrTestFold][files[tuneFoldOrTestFold].files[1]]
            exampleNames_tuneOrTest = files[tuneFoldOrTestFold][files[tuneFoldOrTestFold].files[2]]
        elif dataX_train is None and j != testFold:
            dataX_train = bootstraps[j][i][:,:13]
            dataY_train = bootstraps[j][i][:, [13]]
            exampleNames_train = bootstraps[j][i][:,[14]]  
        elif j != testFold:
            dataX_train = np.vstack([dataX_train, bootstraps[j][i][:,:13]])  
            dataY_train = np.vstack([dataY_train, bootstraps[j][i][:,[13]]])
            exampleNames_train = np.vstack([exampleNames_train, bootstraps[j][i][:,[14]]])  
    tree = createDecisionTree(dataX_train, dataY_train, exampleNames_train, set(), 'rf', depth) 
    # find votes for tune set total 101 decision trees and majority votes  
    positive, negative, res = traverseDescisionTree(dataX_tuneOrTest, dataY_tuneOrTest, tree)
    return positive, negative, res, dataX_tuneOrTest, dataY_tuneOrTest, exampleNames_tuneOrTest

depthPositive = dict.fromkeys(featureCount, 0)
# best tune bootstrap accuracy for each depth
depthBootstrap = dict.fromkeys(featureCount, None)
for depth in featureCount:
    tunSetAcc = None
    tuneFold = None
    for fold in range(9):
        if fold == testFold:
            continue
        tuneFold = fold
        result = [[] for _ in range(files[fold][files[fold].files[0]].shape[0])]
        acc, bootstrapIdx = 0, -1
        for i in range(5):
            # genarate 101 decision trees and take majority votes
            positive, negative, res, dataX_tune, dataY_tune, exampleNames_tune= generateTreesForBootstrap(i, tuneFold, None, None, None)
            if acc < positive:
                acc = positive
                bootstrapIdx = i
            for example in range(len(result)):
                result[example].append(res[example])
        majorityOfVotes = [max(set(x), key=x.count) for x in result]
        # Accuracy of tune set
        positive = 0
        for ex in range(len(dataY_tune)):
            if dataY_tune[ex][0] == majorityOfVotes[ex]:
                positive = positive + 1
        accuracy = ((positive / dataX_tune.shape[0]) * 100)
        arr = [accuracy, tuneFold, bootstrapIdx]
        if tunSetAcc is None:
            tunSetAcc = np.array(arr)
        else:
            tunSetAcc = np.vstack([tunSetAcc, arr])
    depthPositive[depth] = mean(tunSetAcc[:, 0])
    bestTuneForDepth = np.argmax(tunSetAcc[:, 0])
    depthBootstrap[depth] = tunSetAcc[bestTuneForDepth]
print(depthPositive)
print(depthBootstrap)

# d = random.choice(featureCount)
d = 7
print("------Depth------")
print(d)
bootstrapIndex = depthBootstrap[d][2]

positive, negative, res, dataX_test, dataY_test, exampleNames_test = generateTreesForBootstrap(int(bootstrapIndex), int(testFold), None, None, None)
print(f'-----------Unused Test set---------')
print((positive / dataX_test.shape[0]) * 100)

for i in range(len(files)):
    positive, negative, res, dataX_test, dataY_test, exampleNames_test = generateTreesForBootstrap(int(bootstrapIndex), int(i), None, None, None)
    print(f'-----------Test Accuracy {i}---------')
    print
    print((positive / dataX_test.shape[0]) * 100)


