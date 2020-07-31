import torch

def evaluate(set_name, epoch, idx2Label, ground_truth, predictions, writer=None, write_to_tensorboard=True):
    true_positives = (ground_truth == predictions).sum().item()
    accuracy = true_positives / len(ground_truth)

    if write_to_tensorboard:
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
        #if set_name == 'dev':
        #    f1_scores[label] = f1_score
        print(f"\t{idx2Label[label]:<8} | P {precision:.2f} | R {recall:.2f} | F1 {f1_score:.2f}")
        
        if write_to_tensorboard:
            writer.add_scalar(f"Precision/{idx2Label[label]}", precision, epoch + 1)
            writer.add_scalar(f"Recall/{idx2Label[label]}", recall, epoch + 1)
            writer.add_scalar(f"F1Score/{idx2Label[label]}", f1_score, epoch + 1)


