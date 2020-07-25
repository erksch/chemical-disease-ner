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
        tokens = torch.LongTensor([x[0]]).to(device)
        chars = torch.LongTensor([x[1]]).to(device)
        true_labels = torch.LongTensor(Y[i]).to(device)

        predicted_labels = net(tokens, chars)
        predicted_labels = predicted_labels.argmax(axis=2).squeeze(dim=0)

        for j in range(len(true_labels)):
            all_true_labels.append(true_labels[j].item())
            all_predicted_labels.append(predicted_labels[j].item())

    return torch.LongTensor(all_true_labels), torch.LongTensor(all_predicted_labels)

def main(hyperparams={}):
    print(f"Torch Version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    CONFIG = load_config(hyperparams)

    train_writer = SummaryWriter(comment='/train')
    dev_writer = SummaryWriter(comment='/dev')
    test_writer = SummaryWriter(comment='/test')

    print("Preprocessing data...")
    train_sentences = process_dataset_xml(CONFIG['train_set_path'])
    dev_sentences = process_dataset_xml(CONFIG['dev_set_path'])
    test_sentences = process_dataset_xml(CONFIG['test_set_path'])
    print("Done.")

    model_args = {}
    
    if CONFIG['use_pretrained_embeddings']:
        word_embeddings, word2Idx, char2Idx, label2Idx, idx2Label = prepare_embeddings(
            [train_sentences, dev_sentences, test_sentences], 
            embeddings_path=CONFIG['pretrained_embeddings_path'])
        model_args = { 'word_embeddings': torch.FloatTensor(word_embeddings).to(device) }
    else:
        word2Idx, char2Idx, label2Idx, idx2Label = prepare_indices([train_sentences, dev_sentences, test_sentences])
    
    vocab_size = len(word2Idx)
    num_classes = len(label2Idx)
    num_chars = len(char2Idx)
    f1_scores = {label: 0.0 for label in idx2Label.keys()}

    model = BiLSTM(CONFIG, vocab_size=vocab_size, num_classes=num_classes, num_chars=num_chars, **model_args).to(device)

    X_train, Y_train = text_to_indices(train_sentences, word2Idx, char2Idx, label2Idx)

    print("Train dataset class distribution:")
    total = len([token for sentence in Y_train for token in sentence])
    weights = []
    print(f"Total of {total} tokens")
    for i, label in enumerate(label2Idx):
        count = 0
        for sentence in Y_train:
            count += len(np.where(np.array(sentence) == label2Idx[label])[0])
        end = ' | ' if i < len(label2Idx) -1 else ''
        print(f"{label} {count} {count / total:.2f}", end=end)

        weight = CONFIG['non_null_class_weight'] if (count / total) < 0.1 else CONFIG['null_class_weight']
        weights.append(weight)
    print()
    print(f"Weights: {weights}")

    X_dev, Y_dev = text_to_indices(dev_sentences, word2Idx, char2Idx, label2Idx)
    X_test, Y_test = text_to_indices(test_sentences, word2Idx, char2Idx, label2Idx)

    if CONFIG['batch_mode'] == 'padded_sentences':
        dataset_args = { 'pad_sentences': True, 'pad_sentences_max_length': CONFIG['padded_sentences_max_length'] }
    else:
        dataset_args = {}

    dataset = CDRDataset(X_train, Y_train, word2Idx, char2Idx, label2Idx, **dataset_args)

    if CONFIG['batch_mode'] == 'single':
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    elif CONFIG['batch_mode'] == 'padded_sentences':
        dataloader = DataLoader(dataset, batch_size=CONFIG['padded_sentences_batch_size'], shuffle=True)
    elif CONFIG['batch_mode'] == 'by_sentence_length':
        dataloader = DataLoader(dataset, batch_sampler=UniqueSentenceLengthSampler(dataset))

    loss_args = { "weight": torch.FloatTensor(weights).to(device) } if CONFIG['use_weighted_loss'] else {}
    criterion = nn.CrossEntropyLoss(**loss_args)

    if CONFIG['optimizer'] == 'adam':
        print(f"Using Adam optimizer with learning rate {CONFIG['learning_rate']}")
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    elif CONFIG['optimizer'] == 'sgd':
        print(f"Using SGD optimizer with learning rate {CONFIG['learning_rate']} and momentum {CONFIG['momentum']}")
        optimizer = torch.optim.SGD(model.parameters(), lr=CONFIG['learning_rate'], momentum=CONFIG['momentum'])
    else:
        raise Error("No optimizer specified")

    n_iter = 0

    print("Training...")

    for epoch in range(CONFIG['epochs']):  
        epoch_start = time.time()
        model.train()

        for batch_x_tokens, batch_x_chars, batch_y in dataloader:
            optimizer.zero_grad()
            prediction = model(batch_x_tokens, batch_x_chars).reshape(-1, num_classes)
            loss = criterion(prediction, batch_y.reshape(-1))

            loss.backward()
            optimizer.step()

            train_writer.add_scalar('Loss/train_step', loss, n_iter)
            n_iter += 1

        epoch_end = time.time()

        print(f"Epoch {epoch + 1} | Loss {loss.item():.2f} | Duration {(epoch_end - epoch_start):.2f}s")

        if CONFIG['evaluate_only_at_end']:
            should_evaluate = (epoch + 1) == CONFIG['epochs']
        else:
            should_evaluate = (epoch + 1) == CONFIG['epochs'] or epoch % CONFIG['evaluation_interval'] == 0

        if should_evaluate:
            model.eval()

            with torch.no_grad():
                eval_total_start = time.time()

                for set_name, writer, X, Y in [('train', train_writer, X_train, Y_train)]:#, ('dev', dev_writer, X_dev, Y_dev), ('test', test_writer, X_test, Y_test)]:
                    eval_set_start = time.time()
                    ground_truth, predictions = predict_dataset(X, Y, model)
                    true_positives = (ground_truth == predictions).sum().item()
                    accuracy = true_positives / len(ground_truth)
                    writer.add_scalar(f"Accuracy", accuracy, epoch + 1)
                    
                    print(f"{set_name} set evaluation:")
                    
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
                        f1_scores[label] = f1_score

                        print(f"\t{idx2Label[label]:<8} | P {precision:.2f} | R {recall:.2f} | F1 {f1_score:.2f}")

                        writer.add_scalar(f"Precision/{idx2Label[label]}", precision, epoch + 1)
                        writer.add_scalar(f"Recall/{idx2Label[label]}", recall, epoch + 1)
                        writer.add_scalar(f"F1Score/{idx2Label[label]}", f1_score, epoch + 1)

                    eval_set_end = time.time()
                    print(f"\tTook {(eval_set_end - eval_set_start):.2f}s")

                eval_total_end = time.time()
                print(f"\Total evaluation duration {(eval_total_end - eval_total_start):.2f}s")
    
    #f1_mean = (f1_scores[label2Idx['Disease']] + f1_scores[label2Idx['Chemical']]) / 2
    #return -f1_mean   

if __name__ == '__main__':
    main()
