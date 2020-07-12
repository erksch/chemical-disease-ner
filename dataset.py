from torch.utils.data import Dataset
from utils import process_dataset_xml, format_to_tensors

class CDRDataset(Dataset):

    def __init__(self, sentences, word2Idx, label2Idx):
        unknown_idx = word2Idx['UNKNOWN_TOKEN']
        padding_idx = word2Idx['PADDING_TOKEN']

        self.X = []
        self.Y = []

        null_label = 'O'
        max_sentence_length = 0

        for sentence in sentences:
            max_sentence_length = max(max_sentence_length, len(sentence))

        for sentence in sentences:
            word_indices = []
            label_indices = []

            for word, label in sentence:
                if word in word2Idx:
                    wordIdx = word2Idx[word]
                elif word.lower() in word2Idx:
                    wordIdx = word2Idx[word.lower()]
                else:
                    wordIdx = unknown_idx
                word_indices.append(wordIdx)
                label_indices.append(label2Idx[label])

            while len(word_indices) < max_sentence_length:
                word_indices.append(padding_idx)
                label_indices.append(label2Idx[null_label])

            self.X.append(word_indices)
            self.Y.append(label_indices)

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
