# CRF, bi-LSTM-CRF for Named Entity Recognition

this is a proof of concept for using various CRF solutions for named entity recognition.

## requirements
```
gensim
keras
keras-contrib
tensorflow
numpy
pandas
python-crfsuite
```

## to run feature-engineered CRFsuite CRF:

1. run `pycrfsuite-training.ipynb` to fit model
2. see `results/pyCRF-sample.csv` for sample output

## to run bi-LSTM-CRF

1. run `keras-preprocessing.ipynb` to generate formatted model data
2. run `keras_training.py` to train and save model
3. run `keras-decoding.ipynb` to load saved model and decode test sentences
4. see `results/keras-biLSTM-CRF_sample.csv` for sample output

## data

trained on the ConLL-2002 English NER dataset:

https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus

**NB: convert to utf-8 first, converted csv is in repository**

## preprocessing

see: `preprocessing.ipynb`

1. csv is read
2. word, POS-tag and named entity lists are created by sentence
3. a vocabulary for each input/output type is created
4. sentence words, POS-tags and NE's are integer-indexed as lists
5. data is filtered for only sentences with at least one NE tag
6. data is split into train and test sets
7. all necessary information is saved as numpy binaries

## model and training

see: `keras_training.py`

model inputs: text and pos-tag integer-indexed sequences (padded)

model output: named entity tag integer-indexed sequences (padded)

conLL 2002 model result:

`Accuracy: 96.74%`

## decoding

see: `keras-decoding.ipynb` for code, `results/XXXX-sample.csv` for sample decode

this file decodes test set results into human-readable format.

adjust the number of outputs to see in the following line:

`for sent_idx in range(len(X_test_sents[:500])):` << adjust 500 up or down
