# bLSTM-CRF for metaphor detection

## requirements
```
gensim
keras
keras-contrib
matplotlib
numpy
pandas
tensorflow
```

## data, inputs, outputs

data is from the 000 corpus (*todo: add corpus name)

data are text sentences, lower-cased and stripped of metaphor labels ('@'). each word is integer-indexed according to frequency.

labels are 0/1 sequences, with 1 indicating a metaphor.

## to run:

1. run `preprocessing.ipynb` to generate network-readable data in `npy` format
2. run training scripts (`keras_training-lstm.py` and `keras_training-crf.py`)
3. run `decoding.ipynb` to analyse results

## model/training notes

- different word embedding sizes were not thoroughly explored; paper (citation needed) recommends 100-size embeddings as smallest informative size
- setting Embedding layer as `trainable=True` significantly affected bi-LSTM-CRF accuracy (~5%+)
- non-CRF LSTM model was not tweaked; it is direct copy of bi-LSTM-CRF network with modified final layers

## results:

the bidirectional LSTM network does not identify metaphors well; while raw accuracy is high, in truth the network learns to label all words as non-metaphor (over-generalization). see `sample_results_lstm.csv`:

```
on withheld test sentences:

             precision    recall  f1-score   support

        0.0       0.88      1.00      0.94      4968
        1.0       0.00      0.00      0.00       662

avg / total       0.78      0.88      0.83      5630
```

the bi-LSTM-CRF network does significantly better at identifying metaphors; see `sample_results_crf.csv`:

```
on withheld test sentences:

             precision    recall  f1-score   support

        0.0       0.94      0.95      0.95      4968
        1.0       0.63      0.58      0.60       662

avg / total       0.91      0.91      0.91      5630

metaphor-positive label info
true positives : 384
false positives: 229
false negatives: 278
rcll (tp/tp+fn): 58.00604229607251
prec (tp/tp+fp): 62.64274061990212
```

