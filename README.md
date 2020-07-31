# Chemical & Disease Named Entity Recognition

## Local Setup

### Python Setup

go into the repos directory and create a virtual env

```console
foo@bar:~$ cd chemical-disease-ner
foo@bar:~/chemical-disease-ner$ python3 -m venv venv
```

activate the virtual env

```console
foo@bar:~/chemical-disease-ner$ source venv/bin/activate
```

Install the requirements

```console
foo@bar:~/chemical-disease-ner$ pip install -r requirements.txt
```

### Load the embeddings

```console
foo@bar:~/chemical-disease-ner$ cd embeddings
foo@bar:~/chemical-disease-ner/embeddings$ wget https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioWordVec_PubMed_MIMICIII_d200.vec.bin
```

### Execute model

The models parameters can be adapted in the example.ini file
Then you can just launch a trainings run:

```console
foo@bar:~/chemical-disease-ner$ python main.py
```
