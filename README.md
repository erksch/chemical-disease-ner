# Chemical & Disease Named Entity Recognition

## Setup

Clone the repository and enter the directory

```
git clone git@github.com:erksch/chemical-disease-ner.git
cd chemical-disease-ner
```

Create a python virtual env and activate it

```
python3 -m venv venv
source venv/bin/activate
```

Install requirements

```
pip install -r requirements.txt
```

### Downloading the pretrained embeddings

Using pretrained embeddings is optional and can be configured in the config file but using them yields the best results.
The used BioWordVec embeddings can be downloaded to the embeddings folder with

```
cd embeddings
wget https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioWordVec_PubMed_MIMICIII_d200.vec.bin
```

### Training the model

Copy the `example.ini` and create your own config file.
Then run the training and specify your config file with

```
python main.py --config myconfig.ini
```

### Viewing results in TensorBoard

The training procedure automatically writes to the TensorBoard log directory `runs`.
To open TensorBoard and view the training progress run

```
tensorboard --logdir runs
```

### Colab demo

A demo of a simplified version of our model can be found on [colab](https://colab.research.google.com/drive/1xSJgVzwlnMRBxE7SuWLLOLLCFP0EDrvm?usp=sharing).
