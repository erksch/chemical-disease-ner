import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import CDRDataset
from utils import process_dataset_xml, prepare_embeddings, prepare_indices
from model import BiLSTM

def main():
  print(f"Torch Version: {torch.__version__}")
  print(f"CUDA available: {torch.cuda.is_available()}")

  device = torch.device('cuda')
  writer = SummaryWriter()
  
  use_embeddings = True

  print("Preprocessing data...")
  train_sentences = process_dataset_xml('data/CDR_Data/CDR.Corpus.v010516/CDR_TrainingSet.BioC.xml')
  # test_sentences = process_dataset_xml('data/CDR_Data/CDR.Corpus.v010516/CDR_TestSet.BioC.xml')
  # dev_sentences = process_dataset_xml('data/CDR_Data/CDR.Corpus.v010516/CDR_DevelopmentSet.BioC.xml')
  print("Done.")

  if use_embeddings:
      word_embeddings, word2Idx, label2Idx, idx2Label = prepare_embeddings(
            train_sentences, 
            embeddings_dim=200, 
            embeddings_path='embeddings/BioWordVec_PubMed_MIMICIII_d200.vec.bin')
   
      model = BiLSTM(
          vocab_size=len(word2Idx),
          word_embeddings=torch.FloatTensor(word_embeddings).to(device), 
          use_pretrained_embeddings=True,
          num_classes=len(label2Idx)).to(device)
  else:
     word2Idx, label2Idx, idx2Label = prepare_indices(train_sentences)
     model = BiLSTM(
          vocab_size=len(word2Idx),
          use_pretrained_embeddings=False,
          num_classes=len(label2Idx)).to(device)

  num_classes = len(label2Idx)
  epochs = 100
  learning_rate = 0.015
  momentum = 0.9
  batch_size = 100

  dataset = CDRDataset(train_sentences, word2Idx, label2Idx)
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
  n_iter = 0

  print("Training...")

  for epoch in range(epochs):  
        model.train()

        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()

            prediction = model(batch_x).reshape(-1, num_classes)
            loss = criterion(prediction, batch_y.reshape(-1))

            loss.backward()
            optimizer.step()

            writer.add_scalar('Loss/train_step', loss, n_iter)
            n_iter += 1
       
        model.eval()

        """
        ground_truth = Y_train.reshape(-1)
        predictions = model(X_train).argmax(dim=2).reshape(-1)
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

            if true_positives + false_positives == 0:
                precision = 0
            else:
                precision = true_positives / (true_positives + false_positives)

            f1_score = (2 * true_positives) / (2 * true_positives + false_positives + false_negatives)

            print(f"{idx2Label[label]}: Precision {precision}, Recall {recall}, F1 Score {f1_score}")

            writer.add_scalar(f"Precision/{idx2Label[label]}/train_epoch", precision, epoch)
            writer.add_scalar(f"Recall/{idx2Label[label]}/train_epoch", recall, epoch)
            writer.add_scalar(f"F1Score/{idx2Label[label]}/train_epoch", f1_score, epoch)
       """

if __name__ == '__main__':
    main()
