import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import CDRDataset, UniqueSentenceLengthSampler
from config import load_config
from utils import process_dataset_xml, prepare_embeddings, prepare_indices, text_to_indices
from model import BiLSTM

device = torch.device('cuda')

def predict_dataset(X, Y, net):
    all_true_labels = []
    all_predicted_labels = []

    for i, x in enumerate(X):
        tokens = torch.LongTensor([x]).to(device)
        true_labels = torch.LongTensor(Y[i]).to(device)

        predicted_labels = net(tokens)
        predicted_labels = predicted_labels.argmax(axis=2).squeeze(dim=0)

        for j in range(len(true_labels)):
            all_true_labels.append(true_labels[j].item())
            all_predicted_labels.append(predicted_labels[j].item())

    return torch.LongTensor(all_true_labels), torch.LongTensor(all_predicted_labels)

def main():
    print(f"Torch Version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    CONFIG = load_config()

    writer = SummaryWriter()

    print("Preprocessing data...")
    train_sentences = process_dataset_xml('data/CDR_Data/CDR.Corpus.v010516/CDR_TrainingSet.BioC.xml')
    # test_sentences = process_dataset_xml('data/CDR_Data/CDR.Corpus.v010516/CDR_TestSet.BioC.xml')
    # dev_sentences = process_dataset_xml('data/CDR_Data/CDR.Corpus.v010516/CDR_DevelopmentSet.BioC.xml')
    print("Done.")

    model_args = {}
    
    if CONFIG['use_pretrained_embeddings']:
        word_embeddings, word2Idx, label2Idx, idx2Label = prepare_embeddings(
            train_sentences, 
            embeddings_path=CONFIG['pretrained_embeddings_path'])
        model_args = { 'word_embeddings': torch.FloatTensor(word_embeddings).to(device) }
    else:
        word2Idx, label2Idx, idx2Label = prepare_indices(train_sentences)
    
    vocab_size = len(word2Idx)
    num_classes = len(label2Idx)

    model = BiLSTM(CONFIG, vocab_size=vocab_size, num_classes=num_classes, **model_args).to(device)

    X_train, Y_train = text_to_indices(train_sentences, word2Idx, label2Idx)
    dataset = CDRDataset(X_train, Y_train, word2Idx, label2Idx, pad_sentences=(CONFIG['batch_mode'] == 'padded_sentences'))

    if CONFIG['batch_mode'] == 'single':
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    elif CONFIG['batch_mode'] == 'padded_sentences':
        dataloader = DataLoader(dataset, batch_size=CONFIG['padded_sentences_batch_size'], shuffle=True)
    elif CONFIG['batch_mode'] == 'by_sentence_length':
        dataloader = DataLoader(dataset, batch_sampler=UniqueSentenceLengthSampler(dataset))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=CONFIG['learning_rate'], momentum=CONFIG['momentum'])
    n_iter = 0

    print("Training...")

    for epoch in range(CONFIG['epochs']):  
        epoch_start = time.time()
        model.train()

        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()

            prediction = model(batch_x).reshape(-1, num_classes)
            loss = criterion(prediction, batch_y.reshape(-1))

            loss.backward()
            optimizer.step()

            writer.add_scalar('Loss/train_step', loss, n_iter)
            n_iter += 1

        epoch_end = time.time()

        print(f"Epoch {epoch} | Loss {loss.item():.2f} | Duration {(epoch_end - epoch_start):.2f}s")

        model.eval()

        with torch.no_grad():
            eval_start = time.time()

            ground_truth, predictions = predict_dataset(X_train, Y_train, model)
            true_positives = (ground_truth == predictions).sum().item()
            accuracy = true_positives / len(ground_truth)
            writer.add_scalar('Accuracy/All/train_epoch', accuracy, epoch)
            
            print("Evaluation:")
            
            for label in idx2Label.keys():
                indices_in_class = torch.where(ground_truth == label)[0]
                true_positives = (ground_truth[indices_in_class] == predictions[indices_in_class]).sum().item()
                false_negatives = len(indices_in_class) - true_positives
          
                recall = true_positives / len(indices_in_class)

                indices_predicted_in_class = torch.where(predictions == label)[0]
                false_positives = (ground_truth[indices_predicted_in_class] != predictions[indices_predicted_in_class]).sum().item()

                if true_positives + false_positives == 0:
                    precision = 0
                else:
                    precision = true_positives / (true_positives + false_positives)

                f1_score = (2 * true_positives) / (2 * true_positives + false_positives + false_negatives)

                print(f"\t{idx2Label[label]:<8} | P {precision:.2f} | R {recall:.2f} | F1 {f1_score:.2f}")

                writer.add_scalar(f"Precision/{idx2Label[label]}/train_epoch", precision, epoch)
                writer.add_scalar(f"Recall/{idx2Label[label]}/train_epoch", recall, epoch)
                writer.add_scalar(f"F1Score/{idx2Label[label]}/train_epoch", f1_score, epoch)

            eval_end = time.time()
            print(f"\tEvaluation duration {(eval_end - eval_start):.2f}s")

if __name__ == '__main__':
    main()
