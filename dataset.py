import torch
from torch.utils.data import Dataset, Sampler
from utils import chunks

class CDRDataset(Dataset):

    def __init__(self, X, Y, word2Idx, char2Idx, label2Idx, with_chars=False, pad_sentences=False, pad_sentences_max_length=-1):
        self.X = X
        self.Y = Y
        self.word2Idx = word2Idx
        self.char2Idx = char2Idx
        self.label2Idx = label2Idx
        self.max_length = pad_sentences_max_length
        self.with_chars = with_chars

        if pad_sentences:
            self._pad_sentences()
        
    def _pad_sentences(self):
        pad_token = self.word2Idx['PADDING_TOKEN']
        pad_label = self.label2Idx['O']
        
        added_padding_tokens = 0
        stripped_tokens = 0
        
        padded_word_length = len(self.X[0][1][0])
        padded_chars = [self.char2Idx['PADDING'] for i in range(padded_word_length)]

        if self.max_length == -1:
            for sentence in self.X:
                self.max_length = max(self.max_length, len(sentence))

        print(f"Padding sentences to length {self.max_length} with padding token {pad_token}.")

        new_X = []
        new_Y = []

        for i, sentence in enumerate(self.X):
            tokens, chars = sentence
            y = self.Y[i]
            
            if len(tokens) > self.max_length:
                token_chunks = list(chunks(tokens, self.max_length))
                char_chunks = list(chunks(chars, self.max_length))
                y_chunks = list(chunks(y, self.max_length))

                for j in range(len(token_chunks)):
                    while len(token_chunks[j]) < self.max_length:
                        token_chunks[j].append(pad_token)
                        char_chunks[j].append(padded_chars)
                        y_chunks[j].append(pad_label)
                    new_X.append([token_chunks[j], char_chunks[j]])
                    new_Y.append(y_chunks[j])
            else:
                while len(tokens) < self.max_length:
                    tokens.append(pad_token)
                    chars.append(padded_chars)
                    y.append(pad_label)
                new_X.append([tokens, chars])
                new_Y.append(y)

        self.X = new_X
        self.Y = new_Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.with_chars:
            return torch.LongTensor(self.X[idx][0]).to('cuda'), torch.LongTensor(self.X[idx][1]).to('cuda'), torch.LongTensor(self.Y[idx]).to('cuda')
        else:
            return torch.LongTensor(self.X[idx]).to('cuda'), torch.LongTensor(self.Y[idx]).to('cuda')


class UniqueSentenceLengthSampler(Sampler):
    
    def __init__(self, data_source):
        self.data_source = data_source
        self.sentence_lengths = set()

        for x, _ in data_source:
            self.sentence_lengths.add(len(x))

    def __iter__(self):
        batch = []
        for sentence_length in self.sentence_lengths:
            for idx in range(len(self.data_source)):
                x, _ = self.data_source[idx]
                if (len(x) == sentence_length):
                    batch.append(idx)
            yield batch
            batch = []

    def __len__(self):
        return len(self.data_source)


