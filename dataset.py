import torch
from torch.utils.data import Dataset, Sampler

class CDRDataset(Dataset):

    def __init__(self, X, Y, word2Idx, label2Idx, pad_sentences=False, pad_sentences_max_length=-1):
        self.X = X
        self.Y = Y
        self.word2Idx = word2Idx
        self.label2Idx = label2Idx
        self.max_length = pad_sentences_max_length
        
        if pad_sentences:
            self._pad_sentences()
        
    def _pad_sentences(self):
        pad_token = self.word2Idx['PADDING_TOKEN']
        pad_label = self.label2Idx['O']
        added_padding_tokens = 0
        stripped_tokens = 0

        if self.max_length == -1:
            for sentence in self.X:
                self.max_length = max(self.max_length, len(sentence))

        print(f"Padding sentences to length {self.max_length} with padding token {pad_token}.")

        for i, sentence in enumerate(self.X):
            if len(sentence) > self.max_length:
                stripped_tokens += len(sentence) - self.max_length
                self.X[i] = self.X[i][:self.max_length]
                self.Y[i] = self.Y[i][:self.max_length]

            while len(sentence) < self.max_length:
                added_padding_tokens += 1
                self.X[i].append(pad_token)
                self.Y[i].append(pad_label)

        print(f"Removed {stripped_tokens} tokens from sentence longer than {self.max_length}")
        print(f"Added {added_padding_tokens} padding tokens.")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
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


