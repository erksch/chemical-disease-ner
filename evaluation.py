import time
from prediction import predict_dataset

def evaluate(model, word2Idx, char2Idx, CONFIG):
    model.eval()
    prediciton_args = {}
    if CONFIG['use_char_inputs']:
        prediction_args = { 'with_chars': True, pad_chars_to: CONFIG['char_pad_size'] }

    with torch.no_grad():
        eval_total_start = time.time()

        for set_name, writer, X, Y in [('train', train_writer, X_train, Y_train), ('dev', dev_writer, X_dev, Y_dev), ('test', test_writer, X_test, Y_test)]:
            eval_set_start = time.time()
            ground_truth, predictions = predict_dataset(X, Y, model, word2Idx, char2Idx, **prediction_args)
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
                if set_name == 'dev':
                    f1_scores[label] = f1_score

                print(f"\t{idx2Label[label]:<8} | P {precision:.2f} | R {recall:.2f} | F1 {f1_score:.2f}")

                writer.add_scalar(f"Precision/{idx2Label[label]}", precision, epoch + 1)
                writer.add_scalar(f"Recall/{idx2Label[label]}", recall, epoch + 1)
                writer.add_scalar(f"F1Score/{idx2Label[label]}", f1_score, epoch + 1)

            eval_set_end = time.time()
            print(f"\tTook {(eval_set_end - eval_set_start):.2f}s")

        eval_total_end = time.time()
        print(f"\Total evaluation duration {(eval_total_end - eval_total_start):.2f}s")