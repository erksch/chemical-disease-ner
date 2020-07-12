 
import torch
import torch.nn as nn

DROPOUT = 0.5
HIDDEN_DIM = 200
EMBEDDING_DIM = 100

class BiLSTM(nn.Module):
    
    def __init__(self, vocab_size, num_classes, use_pretrained_embeddings=True, word_embeddings=None):
        super(BiLSTM, self).__init__()

        if use_pretrained_embeddings:
            self.embedding_dim = word_embeddings.shape[1] 
            self.embedding = nn.Embedding.from_pretrained(word_embeddings)
        else:
            self.embedding_dim = EMBEDDING_DIM
            self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        
        self.dropout = nn.Dropout(DROPOUT)
        self.lstm = nn.LSTM(self.embedding_dim, HIDDEN_DIM, bidirectional=True)
        self.linear = nn.Linear(HIDDEN_DIM * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.linear(x)

        return x
