# bi-LSTM-CRF for Named Entity Recognition

this is a proof of concept for using LSTM-CRF for named entity recognition.

## requirements
```
gensim
keras
keras-contrib
tensorflow
numpy
pandas
```

## to run

1. add /embeddings and /model to directory
2. run `preprocessing.ipynb` to generate data
3. run `keras_training.py` to train and save model
4. run `decoding.ipynb` to load saved model and decode test sentences

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

model:
```
MAX_LENGTH = 30			# max sent length supported (words)
MAX_VOCAB = 25000 		# out of 29341
EMBEDDING_SIZE = 100	# gensim word2vec embedding size
HIDDEN_SIZE = 200		# LSTM feature size
BATCH_SIZE = 64
DROPOUTRATE = 0.2
MAX_EPOCHS = 4

text layers   : dense embedding > dropout > bi-LSTM
pos layers    : dense embedding > dropout > bi-LSTM
merged layers : concatenate text and pos outputs > bi-LSTM > bi-LSTM > CRF

optimizer='adam'
```

result:

`Accuracy: 97.62% `

## decoding

see: `decoding.ipynb` for code, `test_decode_sample.csv` for sample decode

this file decodes test set results into human-readable format.

adjust the number of outputs to see in the following line:

`for sent_idx in range(len(X_test_sents[:500])):` << adjust 500 up or down
