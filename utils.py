# From: https://github.com/kamalkraj/Named-Entity-Recognition-with-Bidirectional-LSTM-CNNs

import numpy as np

def createBatches(sentences):
    sentence_lengths = []
    for tokens, labels in sentences:
        sentence_lengths.append(len(tokens))
    sentence_lengths = set(sentence_lengths)
    print(f"Unique sentence lengths: {sentence_lengths}")
    print(f"Amount of unique sentence lengths: {len(sentence_lengths)}")
    batches = [] # a batch is a list of sentences with the same length
    batch_len = []
    z = 0
    for i in sentence_lengths:
        for sentence in sentences:
            if len(sentence[0]) == i:
                batches.append(sentence)
                z += 1
        batch_len.append(z)
    return batches, batch_len


# returns matrix with 1 entry = list of 2 elements:
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
