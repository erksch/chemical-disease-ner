import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    
    def __init__(self, CONFIG, vocab_size, num_classes, word_embeddings=None):
        super(BiLSTM, self).__init__()
        
        self.use_dropout = CONFIG['use_dropout']

        if CONFIG['use_pretrained_embeddings']:            
            self.embedding_dim = word_embeddings.shape[1] 
            self.embedding = nn.Embedding.from_pretrained(word_embeddings)
        else:
            self.embedding_dim = CONFIG['embeddings_dim']
            self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        
        self.dropout = nn.Dropout(CONFIG['dropout'])
        self.lstm = nn.LSTM(self.embedding_dim, CONFIG['hidden_dim'], bidirectional=True)
        self.linear = nn.Linear(CONFIG['hidden_dim'] * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.linear(x)

        return x
