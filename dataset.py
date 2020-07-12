import torch
from torch.utils.data import Dataset, Sampler

class CDRDataset(Dataset):

    def __init__(self, X, Y, pad_sentences=True):
        self.X = X
        self.Y = Y
        
        if pad_sentences:
            self._pad_sentences()
        
    def _pad_sentences(self):
        pad_token = 1 # word2Idx['PADDING_TOKEN']
        pad_label = 'O'
        max_sentence_length = 0

        for sentence in self.X:
            max_sentence_length = max(max_sentence_length, len(sentence))

        print(f"Padding sentences to length {max_sentence_length} with padding token {pad_token}.")

        for i, sentence in enumerate(self.X):
            while len(sentence) < max_sentence_length:
                self.X[i].append(pad_token)
                self.Y[i].append(pad_label)

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


