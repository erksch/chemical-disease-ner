# From: https://github.com/kamalkraj/Named-Entity-Recognition-with-Bidirectional-LSTM-CNNs

import numpy as np

def createBatches(data):
    l = []
    for i in data:
        l.append(len(i[0]))
    l = set(l)
    batches = []
    batch_len = []
    z = 0
    for i in l:
        for batch in data:
            if len(batch[0]) == i:
                batches.append(batch)
                z += 1
        batch_len.append(z)
    return batches, batch_len


# returns matrix with 1 entry = list of 4 elements:
# word indices, label indices
def createMatrices(sentences, word2Idx, label2Idx):
    unknownIdx = word2Idx['UNKNOWN_TOKEN']
    paddingIdx = word2Idx['PADDING_TOKEN']

    dataset = []

    wordCount = 0
    unknownWordCount = 0

    for sentence in sentences:
        wordIndices = []
        labelIndices = []

        for word, label in sentence:
            wordCount += 1
            if word in word2Idx:
                wordIdx = word2Idx[word]
            elif word.lower() in word2Idx:
                wordIdx = word2Idx[word.lower()]
            else:
                wordIdx = unknownIdx
                unknownWordCount += 1
            # Get the label and map to int
            wordIndices.append(wordIdx)
            labelIndices.append(label2Idx[label])

        dataset.append([wordIndices, labelIndices])

    return dataset


def iterate_minibatches(dataset, batch_len):
    start = 0
    for i in batch_len:
        tokens = []
        labels = []
        data = dataset[start:i]
        start = i
        for dt in data:
            t, l = dt
            l = np.expand_dims(l, -1)
            tokens.append(t)
            labels.append(l)
        
        yield np.asarray(labels), np.asarray(tokens)
