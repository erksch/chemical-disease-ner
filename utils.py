import time
import torch
import numpy as np
import gensim
from xml.dom import minidom
from nltk.tokenize import sent_tokenize, word_tokenize

def get_text(node):
    return node.childNodes[0].data
    
def process_dataset_xml(file_path):
    xml = minidom.parse(file_path)
    documents = xml.getElementsByTagName('document')
    all_sentences = []
    
    for document in documents:
        text = ""
        passages = document.getElementsByTagName('passage')
        assert(len(passages) == 2)
        title, abstract = passages
        text += get_text(title.getElementsByTagName('text')[0])
        text += ' '
        text += get_text(abstract.getElementsByTagName('text')[0])
        annotations = document.getElementsByTagName('annotation')
        sentences = sent_tokenize(text)
        tokens = [word_tokenize(sentence) for sentence in sentences]
        
        labels = []
        
        for annotation in annotations:
            entity = get_text(annotation.getElementsByTagName('infon')[0])
            location = annotation.getElementsByTagName('location')[0]
            offset = int(location.attributes['offset'].value)
            length = int(location.attributes['length'].value)
            labels.append([text[offset:offset+length], entity])
            
        token_labels = []
        label_idx = 0
        label_start = 0
        
        for sentence in tokens:
            out = []
            
            for token in sentence:
                if label_idx == len(labels):
                    out.append([token, 'O'])
                    continue
                    
                text, entity = labels[label_idx]
                text = text[label_start:]
                
                if token == text:
                    label_idx += 1
                    out.append([token, entity])
                    label_start = 0
                elif text.startswith(token):
                    label_start += len(token)
                    out.append([token, entity])
                elif text in token:
                    label_idx += 1
                    out.append([token, entity])
                else:
                    out.append([token, 'O'])
                    label_start = 0
            
            token_labels.append(out)

        for sentence in token_labels:
            all_sentences.append(sentence)

    return all_sentences

def extract_words_and_labels(datasets):
    labels = set()
    words = set()

    print("Extracting words and labels...")
    for dataset in datasets
        for sentence in dataset:
            for token, label in sentence:
                labels.add(label)
                words.add(token.lower())
    print(f"Extracted {len(words)} words and {len(labels)} labels.")

    return words, labels

def prepare_indices(datasets):
    dataset_vocab, labels = extract_words_and_labels(datasets)

    # mapping for words
    word2Idx = {}
    word2Idx["PADDING_TOKEN"] = 0
    word2Idx["UNKNOWN_TOKEN"] = 1

    for word in dataset_vocab:
        word2Idx[word] = len(word2Idx)

    # mapping for labels
    label2Idx = {}
    for label in labels:
        label2Idx[label] = len(label2Idx)
    
    idx2Label = {v: k for k, v in label2Idx.items()}

    return word2Idx, label2Idx, idx2Label

def prepare_embeddings(datasets, embeddings_path):
    dataset_vocab, labels = extract_words_and_labels(datasets)

    label2Idx = {}
    for label in labels:
        label2Idx[label] = len(label2Idx)
    
    idx2Label = {v: k for k, v in label2Idx.items()}

    word2Idx = {} 
    word_embeddings = []

    print("Loading embeddings...") 
    start = time.time()
    embeddings = gensim.models.KeyedVectors.load_word2vec_format(embeddings_path, binary=True)
    end = time.time()
    print(f"Completed in {end - start} seconds.")
    embeddings_dim = len(embeddings.wv[list(embeddings.vocab.keys())[0]])

    word2Idx["PADDING_TOKEN"] = 0
    vector = np.zeros(embeddings_dim)
    word_embeddings.append(vector)

    word2Idx["UNKNOWN_TOKEN"] = 1
    vector = np.random.uniform(-0.25, 0.25, embeddings_dim)
    word_embeddings.append(vector)

    for word in embeddings.vocab:
        if word in dataset_vocab:
            vector = embeddings.wv[word]
            word_embeddings.append(vector)
            word2Idx[word] = len(word2Idx)

    word_embeddings = np.array(word_embeddings)
    print(f"Found embeddings for {word_embeddings.shape[0]} of {len(dataset_vocab)} words.")
    
    return word_embeddings, word2Idx, label2Idx, idx2Label

def text_to_indices(sentences, word2Idx, label2Idx):
    unknown_idx = word2Idx['UNKNOWN_TOKEN']
    padding_idx = word2Idx['PADDING_TOKEN']

    X = []
    Y = []

    null_label = 'O'

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

        X.append(word_indices)
        Y.append(label_indices)

    return X, Y

