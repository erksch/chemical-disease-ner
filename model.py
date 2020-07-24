import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTM(nn.Module):
    
    def __init__(self, CONFIG, vocab_size, num_classes, word_embeddings=None):
        super(BiLSTM, self).__init__()
        
        self.use_dropout = CONFIG['use_dropout']
        self.use_additional_linear_layers = CONFIG['use_additional_linear_layers']

        if CONFIG['use_pretrained_embeddings']:            
            self.embedding_dim = word_embeddings.shape[1] 
            self.embedding = nn.Embedding.from_pretrained(word_embeddings)
        else:
            self.embedding_dim = CONFIG['embeddings_dim']
            self.embedding = nn.Embedding(vocab_size, self.embedding_dim)

        self.dropout = nn.Dropout(CONFIG['dropout'])
        self.lstm = nn.LSTM(self.embedding_dim, CONFIG['hidden_dim'], bidirectional=True)
        
        # optional additional hidden Layers
        self.linear1 = nn.Linear(CONFIG['hidden_dim'] * 2, CONFIG['hidden_dim'])
        self.linear2 = nn.Linear(CONFIG['hidden_dim'] , 64)
        self.linear3 = nn.Linear(64, num_classes)
            
        self.linear = nn.Linear(CONFIG['hidden_dim'] * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        if self.use_dropout:
            x = self.dropout(x)
        if self.use_additional_linear_layers:
            x = F.relu(self.linear1(x))
            x = F.relu(self.linear2(x))
            x = self.linear3(x)
        else: 
            x = self.linear(x)

        return x
