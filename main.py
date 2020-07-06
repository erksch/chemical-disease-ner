import tensorflow as tf
import keras
from keras.optimizers import Nadam
from process_dataset import process_dataset_xml
from model import Network

print(f"Tensorflow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")

print("Preprocessing data...")
train_data = process_dataset_xml('data/CDR_Data/CDR.Corpus.v010516/CDR_TrainingSet.BioC.xml')
test_data = process_dataset_xml('data/CDR_Data/CDR.Corpus.v010516/CDR_TestSet.BioC.xml')
print("Done.")


"""Set parameters"""

EPOCHS = 30               # paper: 80
DROPOUT = 0.5             # paper: 0.68
DROPOUT_RECURRENT = 0.25  # not specified in paper, 0.25 recommended
LSTM_STATE_SIZE = 200     # paper: 275
LEARNING_RATE = 0.0105    # paper 0.0105
OPTIMIZER = Nadam()       # paper uses SGD(lr=self.learning_rate), Nadam() recommended

"""Construct and run model"""

cnn_blstm = Network(EPOCHS, DROPOUT, DROPOUT_RECURRENT, LSTM_STATE_SIZE, LEARNING_RATE, OPTIMIZER)
cnn_blstm.loadData(train_data, test_data)
cnn_blstm.embed()
cnn_blstm.createBatches()
cnn_blstm.buildModel()
cnn_blstm.train()
