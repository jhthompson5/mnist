import pandas as pd
import numpy as np

def processData(df):
    features = list(df.columns)
    if 'label' in features:
        features.remove('label')
    inputs = df[features]
    images = []
    for i in range(len(inputs["pixel0"])):
        img = []
        bff = []
        for key in inputs.columns:
            index = int(key[5:])
            if index%28 == 0 and index != 0:
                img.append(bff)
                bff = []
            bff.append(inputs[key].iloc[i])
        img.append(bff)
        images.append(img)
    return images

def getLabels(df):
    labels = list(df["label"])
    return labels

def processTestBatch(data):
    testData = processData(data)
    ids = data.index.tolist()
    testing = {
        'X' : np.array(testData),
        'ids':np.array(ids)
    }
    return testing

def processTrainingBatch(data):
    trainingData = processData(data)
    labels = getLabels(data)

    training = {
        'X' : np.array(trainingData),
        'Y' : np.array(labels)
    }
    return training