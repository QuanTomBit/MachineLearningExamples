
# Machine Learning Imports
import pandas as pd
import sklearn as sk
from sklearn import svm

# Standard Imports
from random import shuffle

testDataLen = 0 # Assigned a value in main() to specify how much data to learn from

'''
    hill-valley.csv
        Instances: 1212
        Highest Community accuracy: ~77%
        Features: 101
            100 floats of elevation-like values
            1 value of 0 or 1 where 0=valley and 1=hill
'''

def main():
    dataFrame = pd.read_csv('hill-valley.csv')
    data = dataFrame.values
    shuffle(data) # In case data is organized in some order, shuffle it, so unseen types of data occur less frequently
    global testDataLen # Allows assignment to global value
    testDataLen = int(len(data)*0.75) # Change size of how much data is tested here

    print('Amount of data to learn from is %d out of %d - (%d%s)' %(testDataLen, len(data), int(testDataLen/len(data) * 100), '%'))

    clf = svm.SVC(gamma=0.001, C=57)

    fitData = getLearnData(data)
    targetData = getTargetData(data)

    clf.fit(fitData, targetData)

    accuracy = testAcc(clf, data[testDataLen:])

    print(accuracy)

def getLearnData(data):
    fitData = []

    for r in data[:testDataLen]:
        row = []
        for i in r[:len(r)-1]:
            row.append(i)
        fitData.append(row)

    return fitData

def getTargetData(data):
    targetData = []

    for r in data[:testDataLen]:
        targetData.append(int(r[len(r)-1]))

    return targetData

def testAcc(clf, data):
    total = 0
    correct = 0
    for r in data:
        total += 1
        landLen = len(r)-1 # -1 to exclude target data pointvin index range below
        # predict() returns array so get first index
        if clf.predict([r[:landLen]])[0] == r[landLen]:
            correct += 1

    return correct / total

if __name__ == '__main__':
    main()