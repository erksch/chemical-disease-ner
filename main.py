import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import CDRDataset, UniqueSentenceLengthSampler
from config import load_config
from utils import process_dataset_xml, prepare_embeddings, prepare_indices, text_to_indices, analyze_label_distribution
from prediction import predict_dataset
from evaluation import evaluate
from model import BiLSTM

device = torch.device('cuda')

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
        word_embeddings, word2Idx, char2Idx, label2Idx, idx2Label = prepare_embeddings(
            [train_sentences, dev_sentences, test_sentences], 
            embeddings_path=CONFIG['pretrained_embeddings_path'])
        model_args['word_embeddings'] = torch.FloatTensor(word_embeddings).to(device)
    else:
        word2Idx, char2Idx, label2Idx, idx2Label = prepare_indices([train_sentences, dev_sentences, test_sentences])
    
    vocab_size = len(word2Idx)
    num_classes = len(label2Idx)
    num_chars = len(char2Idx)
    f1_scores = {label: 0.0 for label in idx2Label.keys()}

    model = BiLSTM(CONFIG, vocab_size=vocab_size, num_classes=num_classes, num_chars=num_chars, **model_args).to(device)

    text_to_indices_args = {}
    if CONFIG['use_char_input']:
        text_to_indices_args = { 'with_chars': True, 'pad_chars_to': CONFIG['char_pad_size'] }

    X_train, Y_train = text_to_indices(train_sentences, word2Idx, char2Idx, label2Idx, **text_to_indices_args)
    X_dev, Y_dev = text_to_indices(dev_sentences, word2Idx, char2Idx, label2Idx, **text_to_indices_args)
    X_test, Y_test = text_to_indices(test_sentences, word2Idx, char2Idx, label2Idx, **text_to_indices_args)

    analyze_label_distribution('Train', Y_train, label2Idx)

    if CONFIG['batch_mode'] == 'padded_sentences':
        dataset_args = { 'pad_sentences': True, 'pad_sentences_max_length': CONFIG['padded_sentences_max_length'] }
    else:
        dataset_args = { 'pad_sentences': False }

    dataset = CDRDataset(X_train, Y_train, word2Idx, char2Idx, label2Idx, **dataset_args)

    if CONFIG['batch_mode'] == 'single':
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    elif CONFIG['batch_mode'] == 'padded_sentences':
        dataloader = DataLoader(dataset, batch_size=CONFIG['padded_sentences_batch_size'], shuffle=True)
    elif CONFIG['batch_mode'] == 'by_sentence_length':
        dataloader = DataLoader(dataset, batch_sampler=UniqueSentenceLengthSampler(dataset))
    
    if CONFIG['use_weighted_loss']:
        weights = [CONFIG['null_class_weight'] if label == 'O' else CONFIG['non_null_class_weight'] for label in label2Idx]
        print(f"Using weighted loss with weights {weights}")
        loss_args = { "weight": torch.FloatTensor(weights).to(device) }
    else:
        loss_args = {}

    criterion = nn.CrossEntropyLoss(**loss_args)

    if CONFIG['optimizer'] == 'adam':
        print(f"Using Adam optimizer. LR {CONFIG['learning_rate']}, WD {CONFIG['weight_decay']}")
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    elif CONFIG['optimizer'] == 'sgd':
        print(f"Using SGD optimizer. LR {CONFIG['learning_rate']}, WD {CONFIG['weight_decay']}, momentum {CONFIG['momentum']}")
        optimizer = torch.optim.SGD(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'], momentum=CONFIG['momentum'])
    else:
        raise Error("No optimizer specified")

    n_iter = 0

    print("Training...")

    train_writer = SummaryWriter(comment='/train')
    dev_writer = SummaryWriter(comment='/dev')
    test_writer = SummaryWriter(comment='/test')

    for epoch in range(CONFIG['epochs']):  
        epoch_start = time.time()
        model.train()

        for batch in dataloader:
            model.zero_grad()
            optimizer.zero_grad()

            if CONFIG['use_char_input']:
                batch_x_tokens, batch_x_chars, batch_y = batch
                prediction = model((batch_x_tokens, batch_x_chars)).reshape(-1, num_classes)
            else:
                batch_x, batch_y = batch
                prediction = model(batch_x).reshape(-1, num_classes)

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

                for set_name, writer, X, Y in [('train', train_writer, X_train, Y_train), ('dev', dev_writer, X_dev, Y_dev), ('test', test_writer, X_test, Y_test)]:
                    print(f"{set_name} set evaluation:")
                    evaluate(X, Y, model, word2Idx, idx2Label, char2Idx, CONFIG, writer, epoch + 1)

                eval_total_end = time.time()
                print(f"\Total evaluation duation {(eval_total_end - eval_total_start):.2f}s")
    
    torch.save(model, 'model.pt')
    
    f1_mean = (f1_scores[label2Idx['Disease']] + f1_scores[label2Idx['Chemical']]) / 2
    return -f1_mean   

if __name__ == '__main__':
    main()
