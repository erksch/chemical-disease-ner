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
from evaluation import evaluate

torch.manual_seed(1)
device = torch.device('cuda')

def predict_dataset(X, Y, net):
    all_true_labels = []
    all_predicted_labels = []

    for i, x in enumerate(X):
        tokens = torch.LongTensor(x).to(device)
        true_labels = torch.LongTensor(Y[i]).to(device)

        _, predicted_labels = net(tokens)

        for j in range(len(true_labels)):
            all_true_labels.append(true_labels[j].item())
            all_predicted_labels.append(predicted_labels[j])
        
        if i % 500 == 0:
            print(i)

    return torch.LongTensor(all_true_labels), torch.LongTensor(all_predicted_labels)

def main(hyperparams={}):
    print(f"Torch Version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    CONFIG = load_config(hyperparams)

    print("Preprocessing data...")
    train_sentences = process_dataset_xml(CONFIG['train_set_path'])
    dev_sentences = process_dataset_xml(CONFIG['dev_set_path'])
    test_sentences = process_dataset_xml(CONFIG['test_set_path'])
    print("Done.")

    model_args = {}
    
    if CONFIG['use_pretrained_embeddings']:
        word_embeddings, word2Idx, label2Idx, idx2Label = prepare_embeddings(
            [train_sentences, dev_sentences, test_sentences], 
            embeddings_path=CONFIG['pretrained_embeddings_path'])
        model_args = { 'word_embeddings': torch.FloatTensor(word_embeddings).to(device) }
    else:
        word2Idx, label2Idx, idx2Label = prepare_indices([train_sentences, dev_sentences, test_sentences])
    
    vocab_size = len(word2Idx)
    num_classes = len(label2Idx)
    f1_scores = {label: 0.0 for label in idx2Label.keys()}
    label2Idx_with_start_stop = label2Idx
    label2Idx_with_start_stop["START"] = len(label2Idx_with_start_stop)
    label2Idx_with_start_stop["STOP"] = len(label2Idx_with_start_stop)

    model = BiLSTM(CONFIG, label2Idx_with_start_stop, vocab_size=vocab_size, **model_args).to(device)

    X_train, Y_train = text_to_indices(train_sentences, word2Idx, label2Idx)

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

    X_dev, Y_dev = text_to_indices(dev_sentences, word2Idx, label2Idx)
    X_test, Y_test = text_to_indices(test_sentences, word2Idx, label2Idx)

    if CONFIG['batch_mode'] == 'padded_sentences':
        dataset_args = { 'pad_sentences': True, 'pad_sentences_max_length': CONFIG['padded_sentences_max_length'] }
    else:
        dataset_args = { 'pad_sentences': False }

    dataset = CDRDataset(X_train, Y_train, word2Idx, label2Idx, **dataset_args)

    if CONFIG['batch_mode'] == 'single':
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    elif CONFIG['batch_mode'] == 'padded_sentences':
        dataloader = DataLoader(dataset, batch_size=CONFIG['padded_sentences_batch_size'], shuffle=True)
    elif CONFIG['batch_mode'] == 'by_sentence_length':
        dataloader = DataLoader(dataset, batch_sampler=UniqueSentenceLengthSampler(dataset))

    loss_args = { "weight": torch.FloatTensor(weights).to(device) } if CONFIG['use_weighted_loss'] else {}
    criterion = nn.CrossEntropyLoss(**loss_args)

    if CONFIG['optimizer'] == 'adam':
        print(f"Using Adam optimizer. LR {CONFIG['learning_rate']}, WD {CONFIG['weight_decay']}")
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    elif CONFIG['optimizer'] == 'sgd':
        print(f"Using SGD optimizer. LR {CONFIG['learning_rate']}, WD {CONFIG['weight_decay']}, momentum {CONFIG['momentum']}")
        optimizer = torch.optim.SGD(model.parameters(), lr=CONFIG['learning_rate'], momentum=CONFIG['momentum'], weight_decay=CONFIG['weight_decay'])
    else:
        raise Error("No optimizer specified")

    n_iter = 0

    print("Initial evaluation:")

    with torch.no_grad():
        ground_truth, predictions = predict_dataset(X_train, Y_train, model)
        evaluate('train', 0, idx2Label, ground_truth, predictions, write_to_tensorboard=False)
    
    print("Training...")

    train_writer = SummaryWriter(comment='/train')
    dev_writer = SummaryWriter(comment='/dev')
    test_writer = SummaryWriter(comment='/test')

    for epoch in range(CONFIG['epochs']):  
        epoch_start = time.time()
        model.train()

        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            model.zero_grad()

            loss = model.neg_log_likelihood(batch_x[0], batch_y[0])

            loss.backward()
            optimizer.step()

            train_writer.add_scalar('Loss/train_step', loss, n_iter)
            n_iter += 1
            if n_iter % 500 == 0:
                print(f"Iteration {n_iter} | Loss {loss}")

        epoch_end = time.time()

        print(f"Epoch {epoch + 1} | Loss {loss.item():.2f} | Duration {(epoch_end - epoch_start):.2f}s")
        
        if CONFIG['evaluate_only_at_end']:
            should_evaluate = (epoch + 1) == CONFIG['epochs']
        else:
            should_evaluate = (epoch + 1) == CONFIG['epochs'] or epoch % CONFIG['evaluation_interval'] == 0
        should_evaluate = True

        if should_evaluate:
            model.eval()

            with torch.no_grad():
                eval_total_start = time.time()

                for set_name, writer, X, Y in [('train', train_writer, X_train, Y_train), ('dev', dev_writer, X_dev, Y_dev), ('test', test_writer, X_test, Y_test)]:
                    eval_set_start = time.time()
                    ground_truth, predictions = predict_dataset(X, Y, model)
                    evaluate(set_name, epoch, idx2Label, ground_truth, predictions, writer=writer)
                    eval_set_end = time.time()
                    print(f"\tTook {(eval_set_end - eval_set_start):.2f}s")

                eval_total_end = time.time()
                print(f"\Total evaluation duration {(eval_total_end - eval_total_start):.2f}s")
        
    #f1_mean = (f1_scores[label2Idx['Disease']] + f1_scores[label2Idx['Chemical']]) / 2
    #return -f1_mean   

if __name__ == '__main__':
    main()
