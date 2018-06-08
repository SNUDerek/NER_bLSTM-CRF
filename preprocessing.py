import codecs, re, random
from collections import Counter
import numpy as np

# function to get vocab, maxvocab
# takes sents : list (tokenized lists of sentences)
# takes maxvocab : int (maximum vocab size incl. UNK, PAD
# takes stoplist : list (words to ignore)
# returns vocab_dict (word to index), inv_vocab_dict (index to word)
def get_vocab(sent_toks, maxvocab=10000, min_count=1, stoplist=[], unk='UNK', pad='PAD', verbose=False):
    # get vocab list
    vocab = [word for sent in sent_toks for word in sent]
    sorted_vocab = sorted(Counter(vocab).most_common(), key=lambda x: x[1], reverse=True)
    sorted_vocab = [i for i in sorted_vocab if i[0] not in stoplist and i[0] != unk]
    if verbose:
        print("total vocab:", len(sorted_vocab))
    sorted_vocab = [i for i in sorted_vocab if i[1] >= min_count]
    if verbose:
        print("vocab over min_count:", len(sorted_vocab))
    # reserve for PAD and UNK
    sorted_vocab = [i[0] for i in sorted_vocab[:maxvocab - 2]]
    vocab_dict = {k: v + 1 for v, k in enumerate(sorted_vocab)}
    vocab_dict[unk] = len(sorted_vocab) + 1
    vocab_dict[pad] = 0
    inv_vocab_dict = {v: k for k, v in vocab_dict.items()}

    return vocab_dict, inv_vocab_dict


# function to convert sents to indexed vectors
# takes list : sents (tokenized sentences)
# takes dict : vocab (word to idx mapping)
# returns list of lists of indexed sentences
def index_sents(sent_tokens, vocab_dict, reverse=False, unk_name='UNK', verbose=False):
    vectors = []
    for sent in sent_tokens:
        sent_vect = []
        if reverse:
            sent = sent[::-1]
        for word in sent:
            if word in vocab_dict.keys():
                sent_vect.append(vocab_dict[word])
            else:  # out of max_vocab range or OOV
                sent_vect.append(vocab_dict[unk_name])
        vectors.append(np.asarray(sent_vect))
    vectors = np.asarray(vectors)
    return vectors


# decode an integer-indexed sequence
# takes indexed_list : one integer-indexedf sentence (list or array)
# takes inv_vocab_dict : dict (index to word)
# returns list of string tokens
def decode_sequence(indexed_list, inv_vocab_dict):
    str = []
    for idx in indexed_list:
        # print(intr)
        str.append(inv_vocab_dict[int(idx)])
    return(str)