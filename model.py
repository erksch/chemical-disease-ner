import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

class BiLSTM(nn.Module):
    
    def __init__(self, CONFIG, vocab_size, num_classes, num_chars, batch_size, word_embeddings=None):
        super(BiLSTM, self).__init__()
        
        self.batch_size = batch_size
        self.use_dropout = CONFIG['use_dropout']
        self.use_additional_linear_layers = CONFIG['use_additional_linear_layers']

        self.hidden_dim = CONFIG['hidden_dim']

        self.char_embedding_dim = 30
        self.char_embedding = nn.Embedding(num_chars, self.char_embedding_dim)
        self.conv2d = nn.Conv2d(30, 30, kernel_size=(1, 1))
        self.maxpool = nn.MaxPool2d(kernel_size=(1, 71), stride=1)

        if CONFIG['use_pretrained_embeddings']:            
            self.word_embedding_dim = word_embeddings.shape[1] 
            self.word_embedding = nn.Embedding.from_pretrained(word_embeddings)
        else:
            self.word_embedding_dim = CONFIG['embeddings_dim']
            self.word_embedding = nn.Embedding(vocab_size, self.word_embedding_dim)

        self.dropout = nn.Dropout(CONFIG['dropout'])
        self.lstm = nn.LSTM(self.word_embedding_dim + self.char_embedding_dim, self.hidden_dim, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(self.hidden_dim * 2, num_classes)
        
        # optional additional hidden Layers
        self.linear1 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim , 64)
        self.linear3 = nn.Linear(64, num_classes)

        self.hidden = self._init_hidden(self.batch_size)    

    def _init_hidden(self, batch_size):
        return (torch.randn(2, batch_size, self.hidden_dim).to('cuda'),
                torch.randn(2, batch_size, self.hidden_dim).to('cuda'))

    def forward(self, xt, xc):
        self.hidden = self._init_hidden(xt.shape[0])

        # char input
        xc = self.char_embedding(xc)    # (B, N, C, 30)
        xc = xc.transpose(1, 3).transpose(2, 3)
        xc = self.conv2d(xc)    # (B, N, C, 128)
        xc = self.maxpool(xc)           # (B, N, C, 64)
        xc = xc.squeeze(dim=3).transpose(1, 2)

        # word / token input
        xt = self.word_embedding(xt)    # (B, N, E)

        x = torch.cat((xt, xc), dim=2)         # (B, N, E + 64)

        x, self.hidden = self.lstm(x, self.hidden)             # (B, N, 2*H)

        if self.use_dropout:
            x = self.dropout(x)
       
        if self.use_additional_linear_layers:
            x = F.relu(self.linear1(x))
            x = F.relu(self.linear2(x))
            x = self.linear3(x)
        else: 
            x = self.linear(x)

        return x
