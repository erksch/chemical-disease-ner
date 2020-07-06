def compute_accuracy(predictions, ground_truth, idx2Label):
    correct = 0
    classes_correct = {}
    total = 0
    classes_total = {}

    label_pred = []
    for i, sentence in enumerate(predictions):
        for j, token_label in enumerate(sentence):
            true_label = ground_truth[i][j]
            total += 1
            if not true_label in classes_total:
                classes_total[true_label] = 0
                classes_correct[true_label] = 0
            classes_total[true_label] += 1
            if token_label == true_label:
                correct += 1
                classes_correct[true_label] += 1

    print(f"All classes: {correct} / {total}, Accuracy {correct / total}")

    for label_idx in list(classes_total.keys()):
        print(f"{idx2Label[label_idx]}: {classes_correct[label_idx]} / {classes_total[label_idx]}, Accuracy {classes_correct[label_idx] / classes_total[label_idx]}")

