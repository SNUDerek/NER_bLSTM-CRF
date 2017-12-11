import codecs, re, random
from collections import Counter
from mlxtend.preprocessing import one_hot
import numpy as np

VOCAB_SIZE = 20000
VOCAB_TAGS = 10
MAX_SEQ_LENGTH = 30

# indexes sentences by vocab frequency list
# reserves 0 for UNKs
# todo: probably shoulda used sklearn.####vectorizer

# USAGE
# first get lists like this:
# sents, classes = dataset.get_lists(sents_filename, classes_filename)
# then run train-test split like this:
# train_X, train_y, test_X, test_y, test_set, class_set = \
#     dataset.get_test_train(sents, classes, trainsize=0.8, max_vocab=50000):

# function to get lists from data
# takes corpus as filename (headlines, articles on alternate lines)
# returns lists of sentence token lists, classes
def get_lists(file_corpus, testing=0):
    if testing == 1:
        print('starting dataset.get_lists()...')
    f_corpus = codecs.open(file_corpus, 'rb', encoding='utf8')
    sents = []
    heads = []
    counter = 0

    for line in f_corpus:
        if counter % 2 == 0:
            heads.append(line.strip('\n').split(' '))
        else:
            sents.append(line.strip('\n').split(' '))
        counter += 1
    return(sents, heads)


def get_texts(file_corpus, testing=0):
    if testing == 1:
        print('starting dataset.get_lists()...')
    f_corpus = codecs.open(file_corpus, 'rb', encoding='utf8')
    sents = []
    heads = []
    counter = 0

    for line in f_corpus:
        if counter % 2 == 0:
            heads.append(line.strip('\n'))
        else:
            sents.append(line.strip('\n'))
        counter += 1
    return(sents, heads)


# function to get vocab, maxvocab
# takes list : sents (tokenized lists)
def get_vocab(sents, maxvocab, stoplist=[], testing=0):
    # get vocab list
    vocab = []
    for sent in sents:
        for word in sent:
            vocab.append(word)

    counts = Counter(vocab) # get counts of each word
    vocab_set = list(set(vocab)) # get unique vocab list
    sorted_vocab = sorted(vocab_set, key=lambda x: counts[x], reverse=True) # sort by counts
    sorted_vocab = [i for i in sorted_vocab if i not in stoplist]
    print("\ntotal vocab size:", len(sorted_vocab), '\n')
    sorted_vocab = sorted_vocab[:maxvocab-2]
    print("\ntrunc vocab size:", len(sorted_vocab), '\n')
    vocab_dict = {k: v+1 for v, k in enumerate(sorted_vocab)}
    vocab_dict['UNK'] = len(sorted_vocab)
    vocab_dict['PAD'] = 0
    inv_vocab_dict = {v: k for k, v in vocab_dict.items()}
    return vocab_dict, inv_vocab_dict


# function to convert sents to vectors
# takes list : sents, dict : vocab
# returns list of vectors (as lists)
def index_sents(sents, vocab, testing=0):
    if testing==1:
        print("starting vectorize_sents()...")
    vectors = []
    # iterate thru sents
    for sent in sents:
        sent_vect = []
        for word in sent:
            if word in vocab.keys():
                idx = vocab[word]
                sent_vect.append(idx)
            else: # out of max_vocab range or OOV
                sent_vect.append(vocab['UNK'])
        vectors.append(sent_vect)
    return(vectors)


# def onehot_vectorize(labels):
#     # https://www.reddit.com/r/MachineLearning/comments/31fk7i/converting_target_indices_to_onehotvector/
#     labels = labels.tolist()
#     labelset = []
#     for sent in labels:
#         for label in sent:
#             if label not in labelset:
#                 labelset.append(label)
#     n_labels = len(list(set(labelset)))
#     vectors = np.eye(n_labels+2)[labels]
#     return(vectors)

def onehot_vectorize(matrix, num):
    result = []
    for vector in matrix:
        a = one_hot(vector.tolist(), dtype='int', num_labels=num)
        result.append(a)
    return np.array(result)


# def onehot_vectorize_sents(sents, vocab, max_vocab, testing=0):
#     if testing==1:
#         print("starting vectorize_sents()...")
#     # get sorted vocab
#     vectors = []
#     # iterate thru sents
#     for sent in sents:
#         sent_vect = []
#         for word in sent:
#             one_hot = []
#             idx = vocab.index(word) # reserve 0 for UNK / OOV
#             for i in range(max_vocab):
#                 if i == idx: # matching
#                     one_hot.append(1)
#                 else:
#                     one_hot.append(0)
#             sent_vect.append(one_hot)
#         vectors.append(sent_vect)
#     if testing==1:
#         print("onehot_vectorize_sents[:10]:", vectors[0])
#     return(vectors)

def decode_seq(sent, vocab):
    str = []
    for intr in sent:
        # print(intr)
        str.append(vocab[int(intr)])
    return(str)


# https://github.com/fchollet/keras/issues/2708
# https://github.com/fchollet/keras/issues/1627
def dataGenerator(batch_size,
                  input_filepath='savedata/',
                  xfile='X_train.npy',
                  yfile='y_train_lex.npy',
                  vocabsize=VOCAB_SIZE,
                  epochsize=300000):

    from keras.utils import np_utils

    i = 0
    X = np.load(input_filepath + xfile)
    y = np.load(input_filepath + yfile)

    while True:
        # add in data reading/augmenting code here
        y_batch = onehot_vectorize(y[i:i + batch_size], vocabsize)
        yield (X[i:i + batch_size], y_batch)
        if i + batch_size >= epochsize:
            i = 0
        else:
            i += batch_size


def dataonehotGenerator(batch_size,
                  input_filepath='savedata/',
                  xfile='X_train.npy',
                  yfile='y_train_lex.npy',
                  vocabsize=VOCAB_SIZE,
                  posvocabsize=VOCAB_TAGS,
                  epochsize=300000):

    i = 0
    X = np.load(input_filepath + xfile)
    y = np.load(input_filepath + yfile)

    while True:
        # add in data reading/augmenting code here
        X_batch = onehot_vectorize(X[i:i + batch_size], vocabsize)
        y_batch = onehot_vectorize(y[i:i + batch_size], posvocabsize)
        yield (X_batch, y_batch)
        if i + batch_size >= epochsize:
            i = 0
        else:
            i += batch_size

# generate for pos model, trained on preds
def dataPosPredGenerator(batch_size,
                         input_filepath='savedata/',
                         xfile='X_train.npy',
                         yfile='y_train_pos.npy',
                         posvocabsize=VOCAB_TAGS,
                         epochsize=300000):

    from keras.utils import np_utils
    from mymodels import get_lexmodel_short

    lexmodel = get_lexmodel_short()
    # lexmodel.compile(loss='categorical_crossentropy', optimizer='adam')

    i = 0
    X = np.load(input_filepath + xfile)
    ypos = np.load(input_filepath + yfile)

    while True:
        # add in data reading/augmenting code here
        y_posbatch = onehot_vectorize(ypos[i:i + batch_size], posvocabsize)
        lexpreds = lexmodel.predict(X[i:i + batch_size])
        yield (lexpreds, y_posbatch)
        if i + batch_size >= epochsize:
            i = 0
        else:
            i += batch_size


# generate for stacked model
def dataStackedGenerator(batch_size,
                  input_filepath='savedata/',
                  xfile='X_train.npy',
                  y_lexfile='y_train_lex.npy',
                  y_posfile='y_train_pos.npy',
                  vocabsize=VOCAB_SIZE,
                  posvocabsize=VOCAB_TAGS,
                  epochsize=300000):

    from keras.utils import np_utils

    i = 0
    X = np.load(input_filepath + xfile)
    ylex = np.load(input_filepath + y_lexfile)
    ypos = np.load(input_filepath + y_posfile)

    while True:
        # add in data reading/augmenting code here
        y_lexbatch = onehot_vectorize(ylex[i:i + batch_size], vocabsize)
        y_posbatch = onehot_vectorize(ypos[i:i + batch_size], posvocabsize)
        yield (X[i:i + batch_size], [y_lexbatch, y_posbatch])
        if i + batch_size >= epochsize:
            i = 0
        else:
            i += batch_size
