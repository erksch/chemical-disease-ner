import csv
import os
import gensim
import numpy as np
from keras.models import Model, load_model
from keras.layers import TimeDistributed, Dense, Embedding, Input, LSTM, Bidirectional
from keras.utils import plot_model

from utils import createMatrices, createBatches, iterate_minibatches
from validation import compute_f1

class Network(object):
    
    def __init__(self, EPOCHS, DROPOUT, DROPOUT_RECURRENT, LSTM_STATE_SIZE, LEARNING_RATE, OPTIMIZER):
        
        self.epochs = EPOCHS
        self.dropout = DROPOUT
        self.dropout_recurrent = DROPOUT_RECURRENT
        self.lstm_state_size = LSTM_STATE_SIZE
        self.learning_rate = LEARNING_RATE
        self.optimizer = OPTIMIZER
        
    def loadData(self, trainSentences, testSentences):
        """Load data and add character information"""
        self.trainSentences = trainSentences
        self.testSentences = testSentences

    def embed(self):
        """Create word- and character-level embeddings"""

        labelSet = set()
        words = {}

        # unique words and labels in data  
        for dataset in [self.trainSentences]:#, self.devSentences, self.testSentences]:
            for sentence in dataset:
                for token, label in sentence:
                    labelSet.add(label)
                    words[token.lower()] = True
                    
        # mapping for labels
        self.label2Idx = {}
        for label in labelSet:
            self.label2Idx[label] = len(self.label2Idx)

        # read GLoVE word embeddings
        word2Idx = {}
        self.wordEmbeddings = []

        word2Idx["PADDING_TOKEN"] = 0
        vector = np.zeros(300)
        self.wordEmbeddings.append(vector)

        word2Idx["UNKNOWN_TOKEN"] = 1
        vector = np.random.uniform(-0.25, 0.25, 300)
        self.wordEmbeddings.append(vector)
        
        embeddings = gensim.models.KeyedVectors.load_word2vec_format('embeddings/BioWordVec_PubMed_MIMICIII_d200.vec.bin', binary=True)

        # loop through each word in embeddings
        for word in embeddings.vocab:
            if word.lower() in words:
                vector = embeddings.wv[word]
                self.wordEmbeddings.append(vector)
                word2Idx[word] = len(word2Idx)

        self.wordEmbeddings = np.array(self.wordEmbeddings)
        
        print("Saving word2Idx to file")
        w = csv.writer(open("word2Idx.csv", "w"))
        for key, val in word2Idx.items():
            w.writerow([key, val])

        self.train_set = createMatrices(self.trainSentences, word2Idx, self.label2Idx)
        self.test_set = createMatrices(self.testSentences, word2Idx, self.label2Idx)

        self.idx2Label = {v: k for k, v in self.label2Idx.items()}

        print("Saving idx2Label to file")
        w = csv.writer(open("idx2Label.csv", "w"))
        for key, val in self.idx2Label.items():
            w.writerow([key, val])
        
    def createBatches(self):
        """Create batches"""
        self.train_batch, self.train_batch_len = createBatches(self.train_set)
        self.test_batch, self.test_batch_len = createBatches(self.test_set)
        
    def tag_dataset(self, dataset, model):
        """Tag data with numerical values"""
        correctLabels = []
        predLabels = []
        for i, data in enumerate(dataset):
            tokens, labels = data
            pred = model.predict(np.asarray([tokens]), verbose=False)[0]
            pred = pred.argmax(axis=-1)  # Predict the classes
            correctLabels.append(labels)
            predLabels.append(pred)
        return predLabels, correctLabels
    
    def buildModel(self):
        """Model layers"""

        # word-level input
        words_input = Input(shape=(None,), dtype='int32', name='words_input')
        words = Embedding(input_dim=self.wordEmbeddings.shape[0], output_dim=self.wordEmbeddings.shape[1], weights=[self.wordEmbeddings],
                          trainable=False)(words_input)

        # concat & BLSTM
        output = Bidirectional(LSTM(self.lstm_state_size, 
                                    return_sequences=True, 
                                    dropout=self.dropout,                        # on input to each LSTM block
                                    recurrent_dropout=self.dropout_recurrent     # on recurrent input signal
                                   ), name="BLSTM")(words)
        output = TimeDistributed(Dense(len(self.label2Idx), activation='softmax'),name="Softmax_layer")(output)

        # set up model
        self.model = Model(inputs=words_input, outputs=[output])
        
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=self.optimizer)
        
        self.init_weights = self.model.get_weights()
        
        plot_model(self.model, to_file='model.png')
        
        print("Model built. Saved model.png\n")
        
    def train(self):
        """Default training"""

        self.f1_test_history = []
        self.f1_dev_history = []

        for epoch in range(self.epochs):    
            print("Epoch {}/{}".format(epoch, self.epochs))
            for i,batch in enumerate(iterate_minibatches(self.train_batch,self.train_batch_len)):
                labels, tokens = batch       
                self.model.train_on_batch([tokens], labels)

            # compute F1 scores
            predLabels, correctLabels = self.tag_dataset(self.test_batch, self.model)
            pre_test, rec_test, f1_test = compute_f1(predLabels, correctLabels, self.idx2Label)
            self.f1_test_history.append(f1_test)
            print("f1 test ", round(f1_test, 4))

            #predLabels, correctLabels = self.tag_dataset(self.dev_batch, self.model)
            #pre_dev, rec_dev, f1_dev = compute_f1(predLabels, correctLabels, self.idx2Label)
            #self.f1_dev_history.append(f1_dev)
            #print("f1 dev ", round(f1_dev, 4), "\n")
            
        print("Final F1 test score: ", f1_test)
            
        print("Training finished.")
            
        # save model
        self.modelName = "{}_{}_{}_{}_{}_{}_{}".format(self.epochs, 
                                                        self.dropout, 
                                                        self.dropout_recurrent, 
                                                        self.lstm_state_size,
                                                        self.conv_size,
                                                        self.learning_rate,
                                                        self.optimizer.__class__.__name__
                                                       )
        
        modelName = self.modelName + ".h5"
        self.model.save(modelName)
        print("Model weights saved.")
        
        self.model.set_weights(self.init_weights)  # clear model
        print("Model weights cleared.")
