import torch
import torch.nn as nn
import torch.nn.functional as F

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class BiLSTM(nn.Module):
    
    def __init__(self, CONFIG, label_to_idx, vocab_size, word_embeddings=None):
        super(BiLSTM, self).__init__()
       
        self.start_label_idx = label_to_idx['START']
        self.stop_label_idx = label_to_idx['STOP']
        self.num_classes = len(label_to_idx)
        self.hidden_dim = CONFIG['hidden_dim']
        self.use_dropout = CONFIG['use_dropout']
        self.use_additional_linear_layers = CONFIG['use_additional_linear_layers']

        if CONFIG['use_pretrained_embeddings']:            
            self.embedding_dim = word_embeddings.shape[1] 
            self.embedding = nn.Embedding.from_pretrained(word_embeddings)
        else:
            self.embedding_dim = CONFIG['embeddings_dim']
            self.embedding = nn.Embedding(vocab_size, self.embedding_dim)

        self.dropout = nn.Dropout(CONFIG['dropout'])
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=1, bidirectional=True)
        self.linear = nn.Linear(self.hidden_dim * 2, self.num_classes)
       
        # optional additional hidden Layers
        self.linear1 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, 64)
        self.linear3 = nn.Linear(64, self.num_classes)

        self.transitions = nn.Parameter(torch.randn(self.num_classes, self.num_classes).to('cuda'))
        self.transitions.data[self.start_label_idx, :] = -10000
        self.transitions.data[:, self.stop_label_idx] = -10000

        self.lstm_hidden_state = self._init_lstm_hidden_state()

    def _init_lstm_hidden_state(self):
        return (torch.randn(2, 1, self.hidden_dim).to('cuda'),
                torch.randn(2, 1, self.hidden_dim).to('cuda'))

    def _get_lstm_features(self, x):
        self.lstm_hidden_state = self._init_lstm_hidden_state()
        sequence_len = len(x)

        x = self.embedding(x).view(sequence_len, 1, -1)
        x, self.lstm_hidden_state = self.lstm(x, self.lstm_hidden_state)
        x = x.view(sequence_len, self.hidden_dim * 2)
        x = self.linear(x)
        return x

    def _forward_alg(self, feats):
        init_alphas = torch.full((1, self.num_classes), -10000.).to('cuda')
        init_alphas[0][self.start_label_idx] = 0.
        forward_var = init_alphas
        
        for feat in feats:
            alphas_t = []
            for next_tag in range(self.num_classes):
                emit_score = feat[next_tag].view(1, -1).expand(1, self.num_classes)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.stop_label_idx]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _score_sentence(self, feats, labels):
        score = torch.zeros(1).to('cuda')
        labels = torch.cat([torch.tensor([self.start_label_idx], dtype=torch.long).to('cuda'), labels])
        for i, feat in enumerate(feats):
            score = score + self.transitions[labels[i + 1], labels[i]] + feat[labels[i + 1]]
        score = score + self.transitions[self.stop_label_idx, labels[-1]]
        return score

    def _viterbi_decode(self, x):
        backpointers = []
        init_vvars = torch.full((1, self.num_classes), -10000.).to('cuda')
        init_vvars[0][self.start_label_idx] = 0

        forward_var = init_vvars
        for feature in x:
            backpointers_t = []
            viterbivars_t = []

            for next_tag in range(self.num_classes):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                backpointers_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))

            forward_var = (torch.cat(viterbivars_t) + feature).view(1, -1)
            backpointers.append(backpointers_t)

        terminal_var = forward_var + self.transitions[self.stop_label_idx]
        best_tag_ud = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        best_path = [best_tag_id]
        for backpointers_t in reversed(backpointers):
            best_tag_id = backpointers_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.start_label_idx
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, x, y):
        feats = self._get_lstm_features(x)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, y)
        return forward_score - gold_score

    def forward(self, x):
        lstm_feats = self._get_lstm_features(x)
        score, labels = self._viterbi_decode(lstm_feats)
        return score, labels
