import torch

def predict_dataset(X, Y, model, word2Idx, char2Idx, with_chars=False, pad_chars_to=None):
    padding_token = word2Idx['PADDING_TOKEN']

    if with_chars:
        padding_char = char2Idx['PADDING']
        padding_token_chars = [padding_char for i in range(pad_chars_to)]

    all_true_labels = []
    all_predicted_labels = []

    for i, x in enumerate(X):
        true_labels = Y[i]

        if with_chars:
            tokens, chars = x
            tokens = torch.LongTensor([tokens]).to(device)
            chars = torch.LongTensor([chars]).to(device)
            predicted_labels = model((tokens, chars))
        else:
            tokens = torch.LongTensor([x]).to(device)
            predicted_labels = model(tokens)
        
        predicted_labels = predicted_labels.argmax(axis=2).squeeze(dim=0)[:num_no_pad_tokens]
            
        for j in range(len(true_labels)):
            all_true_labels.append(true_labels[j])
            all_predicted_labels.append(predicted_labels[j].item())

    return torch.LongTensor(all_true_labels), torch.LongTensor(all_predicted_labels)