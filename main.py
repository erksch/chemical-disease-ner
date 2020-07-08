import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from utils import process_dataset_xml, format_to_tensor, prepare_embeddings
from model import BiLSTM

def main():
  print(f"Torch Version: {torch.__version__}")
  print(f"CUDA available: {torch.cuda.is_available()}")

  writer = SummaryWriter()

  print("Preprocessing data...")
  train_sentences = process_dataset_xml('data/CDR_Data/CDR.Corpus.v010516/CDR_TrainingSet.BioC.xml')
  # test_sentences = process_dataset_xml('data/CDR_Data/CDR.Corpus.v010516/CDR_TestSet.BioC.xml')
  # dev_sentences = process_dataset_xml('data/CDR_Data/CDR.Corpus.v010516/CDR_DevelopmentSet.BioC.xml')
  print("Done.")

  word_embeddings, word2Idx, label2Idx, idx2Label = prepare_embeddings(
      train_sentences, 
      embeddings_dim=200, 
      embeddings_path='embeddings/BioWordVec_PubMed_MIMICIII_d200.vec.bin')
  train_sentences = format_to_tensor(train_sentences, word2Idx, label2Idx)

  model = BiLSTM(word_embeddings=torch.FloatTensor(word_embeddings), num_classes=len(label2Idx))

  epochs = 100
  learning_rate = 0.015
  momentum = 0.9

  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
  n_iter = 0

  def predict_dataset(dataset, net):
    all_true_labels = []
    all_predicted_labels = []

    for tokens, true_labels in dataset:
        tokens = torch.LongTensor([tokens])
        true_labels = torch.LongTensor(true_labels)

        predicted_labels = net(tokens)
        predicted_labels = predicted_labels.argmax(axis=2).squeeze(dim=0)

        for i in range(len(true_labels)):
            all_true_labels.append(true_labels[i].item())
            all_predicted_labels.append(predicted_labels[i].item())

    return torch.LongTensor(all_true_labels), torch.LongTensor(all_predicted_labels)

  print("Training...")

  for epoch in range(epochs):  
        model.train()
    
        for tokens, true_labels in train_sentences:
            tokens = torch.LongTensor([tokens])
            true_labels = torch.LongTensor(true_labels)
            optimizer.zero_grad()
            predicted_labels = model(tokens)
            predicted_labels = predicted_labels.squeeze(dim=0)
            loss = criterion(predicted_labels, true_labels)
            loss.backward()
            optimizer.step()

            writer.add_scalar('Loss/train_step', loss, n_iter)
            n_iter += 1
        

        model.eval()

        ground_truth, predictions = predict_dataset(train_sentences, model)
        true_positives = (ground_truth == predictions).sum().item()
        accuracy = true_positives / len(ground_truth)
        writer.add_scalar('Accuracy/All/train_epoch', accuracy, epoch) 

        print(f"Epoch {epoch}, Loss {loss.item()}, Accuracy {accuracy}")
        
        for label in idx2Label.keys():
            indices_in_class = torch.where(ground_truth == label)[0]
            true_positives = (ground_truth[indices_in_class] == predictions[indices_in_class]).sum().item()
            false_negatives = len(indices_in_class) - true_positives
            
            recall = true_positives / len(indices_in_class)
            
            indices_predicted_in_class = torch.where(predictions == label)[0]
            false_positives = (ground_truth[indices_predicted_in_class] != predictions[indices_predicted_in_class]).sum().item()

            precision = true_positives / (true_positives + false_positives)

            f1_score = (2 * true_positives) / (2 * true_positives + false_positives + false_negatives)

            print(f"{idx2Label[label]}: Precision {precision}, Recall {recall}, F1 Score {f1_score}")

            writer.add_scalar(f"Precision/{idx2Label[label]}/train_epoch", precision, epoch)
            writer.add_scalar(f"Recall/{idx2Label[label]}/train_epoch", recall, epoch)
            writer.add_scalar(f"F1Score/{idx2Label[label]}/train_epoch", f1_score, epoch)
       
if __name__ == '__main__':
    main()
