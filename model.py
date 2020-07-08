 
import torch
import torch.nn as nn

DROPOUT = 0.5
HIDDEN_DIM = 200

class BiLSTM(nn.Module):
    
    def __init__(self, word_embeddings, num_classes):
        super(BiLSTM, self).__init__()

        self.embedding_dim = word_embeddings.shape[1] 

        self.embedding = nn.Embedding.from_pretrained(word_embeddings)
        self.dropout = nn.Dropout(DROPOUT)
        self.lstm = nn.LSTM(self.embedding_dim, HIDDEN_DIM, bidirectional=True)
        self.linear = nn.Linear(HIDDEN_DIM * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.linear(x)

        return x
