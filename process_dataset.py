from xml.dom import minidom
from nltk.tokenize import sent_tokenize, word_tokenize

def get_text(node):
    return node.childNodes[0].data
    
def process_dataset_xml(file_path):
    xml = minidom.parse(file_path)
    documents = xml.getElementsByTagName('document')
    all_sentences = []
    
    for document in documents:
        text = ""
        passages = document.getElementsByTagName('passage')
        assert(len(passages) == 2)
        title, abstract = passages
        text += get_text(title.getElementsByTagName('text')[0])
        text += ' '
        text += get_text(abstract.getElementsByTagName('text')[0])
        annotations = document.getElementsByTagName('annotation')
        sentences = sent_tokenize(text)
        tokens = [word_tokenize(sentence) for sentence in sentences]
        
        labels = []
        
        for annotation in annotations:
            entity = get_text(annotation.getElementsByTagName('infon')[0])
            location = annotation.getElementsByTagName('location')[0]
            offset = int(location.attributes['offset'].value)
            length = int(location.attributes['length'].value)
            labels.append([text[offset:offset+length], entity])
            
        token_labels = []
        label_idx = 0
        label_start = 0
        
        for sentence in tokens:
            out = []
            
            for token in sentence:
                if label_idx == len(labels):
                    out.append([token, 'O'])
                    continue
                    
                text, entity = labels[label_idx]
                text = text[label_start:]
                
                if token == text:
                    label_idx += 1
                    out.append([token, entity])
                    label_start = 0
                elif text.startswith(token):
                    label_start += len(token)
                    out.append([token, entity])
                elif text in token:
                    label_idx += 1
                    out.append([token, entity])
                else:
                    out.append([token, 'O'])
                    label_start = 0
            
            token_labels.append(out)

        for sentence in token_labels:
            all_sentences.append(sentence)

    return all_sentences