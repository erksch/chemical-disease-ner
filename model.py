import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    
    def __init__(self, CONFIG, vocab_size, num_classes, word_embeddings=None):
        super(BiLSTM, self).__init__()
        
        self.use_dropout = CONFIG['use_dropout']
        self.use_additional_linear_layers = CONFIG.['use_additional_linear_layers']

        if CONFIG['use_pretrained_embeddings']:            
            self.embedding_dim = word_embeddings.shape[1] 
            self.embedding = nn.Embedding.from_pretrained(word_embeddings)
        else:
            self.embedding_dim = CONFIG['embeddings_dim']
            self.embedding = nn.Embedding(vocab_size, self.embedding_dim)

        self.dropout = nn.Dropout(CONFIG['dropout'])
        self.lstm = nn.LSTM(self.embedding_dim, CONFIG['hidden_dim'], bidirectional=True)
        
        # optional additional hidden Layers
        
        self.linear1 = nn.Sequential(nn.Linear(CONFIG['hidden_dim'] * 2, CONFIG['hidden_dim']),nn.Sigmoid())
        self.linear2 = nn.Sequential(nn.Linear(CONFIG['hidden_dim'] , num_classes * 10),nn.Sigmoid())
        self.linear3 = nn.Linear(num_classes * 10, num_classes)
            
        self.linear = nn.Linear(CONFIG['hidden_dim'] * 2, num_classes)


    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        if self.use_dropout:
            x = self.dropout(x)
        if self.use_additional_linear_layers:
            x = self.linear1(x)
            x = self.linear2(x)
            x = self.linear3(x)
        else: 
            x = self.linear(x)

        return x
