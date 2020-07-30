from utils import process_dataset_xml

def test():
    train_set = "data/CDR_Data/CDR.Corpus.v010516/CDR_TrainingSet.BioC.xml"
    train_sentences = process_dataset_xml(train_set)

    assert train_sentences[0] == [['Naloxone', 'Chemical'], ['reverses', 'O'], ['the', 'O'], ['antihypertensive', 'O'], ['effect', 'O'], ['of', 'O'], ['clonidine', 'Chemical'], ['.', 'O']]

    assert train_sentences[2] == [['The', 'O'], ['hypotensive', 'Disease'], ['effect', 'O'], ['of', 'O'], ['100', 'O'], ['mg/kg', 'O'], ['alpha-methyldopa', 'Chemical'], ['was', 'O'], ['also', 'O'], ['partially', 'O'], ['reversed', 'O'], ['by', 'O'], ['naloxone', 'Chemical'], ['.', 'O']]

    assert train_sentences[4] == [['In', 'O'], ['brain', 'O'], ['membranes', 'O'], ['from', 'O'], ['spontaneously', 'O'], ['hypertensive', 'Disease'], ['rats', 'O'], ['clonidine', 'Chemical'], [',', 'O'], ['10', 'O'], ['(', 'O'], ['-8', 'O'], [')', 'O'], ['to', 'O'], ['10', 'O'], ['(', 'O'], ['-5', 'O'], [')', 'O'], ['M', 'O'], [',', 'O'], ['did', 'O'], ['not', 'O'], ['influence', 'O'], ['stereoselective', 'O'], ['binding', 'O'], ['of', 'O'], ['[', 'Chemical'], ['3H', 'Chemical'], [']', 'Chemical'], ['-naloxone', 'Chemical'], ['(', 'O'], ['8', 'O'], ['nM', 'O'], [')', 'O'], [',', 'O'], ['and', 'O'], ['naloxone', 'Chemical'], [',', 'O'], ['10', 'O'], ['(', 'O'], ['-8', 'O'], [')', 'O'], ['to', 'O'], ['10', 'O'], ['(', 'O'], ['-4', 'O'], [')', 'O'], ['M', 'O'], [',', 'O'], ['did', 'O'], ['not', 'O'], ['influence', 'O'], ['clonidine-suppressible', 'Chemical'], ['binding', 'O'], ['of', 'O'], ['[', 'Chemical'], ['3H', 'Chemical'], [']', 'Chemical'], ['-dihydroergocryptine', 'Chemical'], ['(', 'O'], ['1', 'O'], ['nM', 'O'], [')', 'O'], ['.', 'O']]

    dev_set = "data/CDR_Data/CDR.Corpus.v010516/CDR_DevelopmentSet.BioC.xml"
    dev_sentences = process_dataset_xml(dev_set)

    assert dev_sentences[0] == [['Tricuspid', 'Disease'], ['valve', 'Disease'], ['regurgitation', 'Disease'], ['and', 'O'], ['lithium', 'Chemical'], ['carbonate', 'Chemical'], ['toxicity', 'Disease'], ['in', 'O'], ['a', 'O'], ['newborn', 'O'], ['infant', 'O'], ['.', 'O']]

    assert dev_sentences[51] == [['Pretreatment', 'O'], ['with', 'O'], ['type-2', 'O'], ['serotonin', 'Chemical'], ['antagonists', 'O'], ['may', 'O'], ['be', 'O'], ['clinically', 'O'], ['useful', 'O'], ['in', 'O'], ['attenuating', 'O'], ['opiate-induced', 'O'], ['rigidity', 'Disease'], [',', 'O'], ['although', 'O'], ['further', 'O'], ['studies', 'O'], ['will', 'O'], ['be', 'O'], ['necessary', 'O'], ['to', 'O'], ['assess', 'O'], ['the', 'O'], ['interaction', 'O'], ['of', 'O'], ['possibly', 'O'], ['enhanced', 'O'], ['CNS', 'O'], [',', 'O'], ['cardiovascular', 'Disease'], [',', 'Disease'], ['and', 'Disease'], ['respiratory', 'Disease'], ['depression', 'Disease'], ['.', 'O']]

if __name__ == '__main__':
    test()
