import torch
from torch.utils.data import Dataset, Sampler

class CDRDataset(Dataset):

    def __init__(self, X, Y, word2Idx, char2Idx, label2Idx, pad_sentences=True, pad_sentences_max_length=-1):
        self.X = X
        self.Y = Y
        self.word2Idx = word2Idx
        self.char2Idx = char2Idx
        self.label2Idx = label2Idx
        self.max_length = pad_sentences_max_length

        if pad_sentences:
            self._pad_sentences()
        
    def _pad_sentences(self):
        pad_token = self.word2Idx['PADDING_TOKEN']
        pad_label = self.label2Idx['O']
        padded_word_length = len(self.X[0][1][0])
        padded_chars = [self.char2Idx['PADDING'] for i in range(padded_word_length)]

        if self.max_length == -1:
            for sentence in self.X:
                self.max_length = max(self.max_length, len(sentence))

        print(f"Padding sentences to length {self.max_length} with padding token {pad_token}.")

        for i, sentence in enumerate(self.X):
            tokens, chars = sentence
            if len(tokens) > self.max_length:
                self.X[i][0] = self.X[i][0][:self.max_length]
                self.X[i][1] = self.X[i][1][:self.max_length]
                self.Y[i] = self.Y[i][:self.max_length]

            while len(tokens) < self.max_length:
                self.X[i][0].append(pad_token)
                self.X[i][1].append(padded_chars)
                self.Y[i].append(pad_label)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.LongTensor(self.X[idx][0]).to('cuda'), torch.LongTensor(self.X[idx][1]).to('cuda'), torch.LongTensor(self.Y[idx]).to('cuda')

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


