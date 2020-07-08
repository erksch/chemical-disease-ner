import torch
import torch.nn as nn
import numpy as np
from utils import process_dataset_xml, format_to_tensor, prepare_embeddings
from model import BiLSTM

def main():
  print(f"Torch Version: {torch.__version__}")
  print(f"CUDA available: {torch.cuda.is_available()}")

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

  epochs = 50
  learning_rate = 0.015
  momentum = 0.9

  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

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

        print(f"Epoch {epoch}, Loss {loss.item()}")

        model.eval()

        correct = 0
        total = 0
        classes_correct = {}
        classes_total = {}

        for tokens, true_labels in train_sentences:
            tokens = torch.LongTensor([tokens])
            true_labels = torch.LongTensor(true_labels)
            predicted_labels = model(tokens)
            predicted_labels = predicted_labels.argmax(axis=2).squeeze(dim=0)
            total += len(predicted_labels)
            correct += (predicted_labels == true_labels).sum().item()

            for i, true_label in enumerate(true_labels):
                true_label = true_label.item()
                if not true_label in classes_correct:
                    classes_correct[true_label] = 0
                    classes_total[true_label] = 0
                classes_total[true_label] += 1 
                if true_label == predicted_labels[i].item():
                    classes_correct[true_label] += 1

        print(f"All classes: {correct} / {total}, Accuracy {correct / total}")

        for label_idx in list(classes_total.keys()):
            print(f"{idx2Label[label_idx]}: {classes_correct[label_idx]} / {classes_total[label_idx]}, Accuracy {classes_correct[label_idx] / classes_total[label_idx]}")    

if __name__ == '__main__':
    main()
