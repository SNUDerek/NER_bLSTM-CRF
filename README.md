# CRF, bi-LSTM-CRF for Named Entity Recognition

this is a proof of concept for using various CRF solutions for named entity recognition. the demos here use all-lower-cased text in order to simulate NER on text where case information is not available (e.g. automatic speech recognition output)

June 08 2018 update:

- now train/test split is uniform across models
- use the `pycrfsuite` report for both models
- added MIT licence for the `pycrfsuite` code
- removed unneeded/unattributed code, trimmed requirements
- expanded comments
- added results

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

1. run `data-preprocessing.ipynb` to generate formatted model data
2. run `pycrfsuite-training.ipynb` to fit model
3. see `results/pyCRF-sample.csv` for sample output

## to run bi-LSTM-CRF

1. run `data-preprocessing.ipynb` to generate formatted model data
2. run `keras_training.ipynb` to train and save model
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

## models and training

see: `pycrfsuite-training.ipynb`

model inputs: word and pos-tag hand-engineered features

model output: named entity tag sequences

see: `keras_training.ipynb`

model inputs: word and pos-tag integer-indexed sequences (padded)

model output: named entity tag integer-indexed sequences (padded)


## decoding

see: `keras-decoding.ipynb` for code, `results/XXXX-sample.csv` for sample decode

this file decodes test set results into human-readable format.

adjust the number of outputs to see in the following line:

`for sent_idx in range(len(X_test_sents[:500])):` << adjust 500 up or down

## performance

per-tag results on the withheld *test set*

### `py-crfsuite`

```
             precision    recall  f1-score   support

      B-art       0.31      0.06      0.10        69
      I-art       0.00      0.00      0.00        54
      B-eve       0.52      0.35      0.42        46
      I-eve       0.35      0.22      0.27        36
      B-geo       0.85      0.90      0.87      5629
      I-geo       0.81      0.74      0.77      1120
      B-gpe       0.94      0.92      0.93      2316
      I-gpe       0.89      0.65      0.76        26
      B-nat       0.73      0.46      0.56        24
      I-nat       0.60      0.60      0.60         5
      B-org       0.78      0.69      0.73      2984
      I-org       0.77      0.76      0.76      2377
      B-per       0.81      0.81      0.81      2424
      I-per       0.81      0.90      0.85      2493
      B-tim       0.92      0.83      0.87      2989
      I-tim       0.82      0.70      0.75      1017

avg / total       0.83      0.82      0.82     23609
```

### `keras biLSTM-CRF`

```
             precision    recall  f1-score   support

      B-art       0.26      0.14      0.18        66
      I-art       0.17      0.07      0.10        54
      B-eve       0.34      0.25      0.29        44
      I-eve       0.20      0.21      0.20        34
      B-geo       0.87      0.90      0.89      5436
      I-geo       0.79      0.83      0.81      1065
      B-gpe       0.96      0.95      0.95      2284
      I-gpe       0.71      0.60      0.65        25
      B-nat       0.58      0.65      0.61        23
      I-nat       1.00      0.40      0.57         5
      B-org       0.80      0.75      0.77      2897
      I-org       0.84      0.77      0.81      2286
      B-per       0.84      0.85      0.84      2396
      I-per       0.84      0.90      0.87      2449
      B-tim       0.90      0.89      0.90      2891
      I-tim       0.84      0.75      0.80       957

avg / total       0.85      0.85      0.85     22912
```