import torch
import time
from prediction import predict_dataset

def evaluate(X, Y, model, word2Idx, idx2Label, char2Idx, CONFIG, writer, writer_step):
    start = time.time()

    prediction_args = {}
    if CONFIG['use_char_input']:
        prediction_args = { 'with_chars': True, 'pad_chars_to': CONFIG['char_pad_size'] }

    ground_truth, predictions = predict_dataset(X, Y, model, word2Idx, char2Idx, **prediction_args)
    true_positives = (ground_truth == predictions).sum().item()
    accuracy = true_positives / len(ground_truth)
    writer.add_scalar(f"Accuracy", accuracy, writer_step)
           
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

        writer.add_scalar(f"Precision/{idx2Label[label]}", precision, writer_step)
        writer.add_scalar(f"Recall/{idx2Label[label]}", recall, writer_step)
        writer.add_scalar(f"F1Score/{idx2Label[label]}", f1_score, writer_step)

    end = time.time()
    print(f"\tTook {(end - start):.2f}s")

