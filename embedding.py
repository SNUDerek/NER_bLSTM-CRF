import codecs
import json
import numpy as np
from gensim.models import Word2Vec
from keras.layers import Embedding


# create embeddings with gensim
def create_embeddings(sentences,
                      embeddings_path='temp_embeddings/embeddings.gensimmodel',
                      vocab_path='temp_embeddings/mapping.json',
                      **params):

    model = Word2Vec(sentences, **params)
    model.save(embeddings_path)
    # weights = model.syn0
    # np.save(open(embeddings_path, 'wb'), weights)
    
    vocab = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    with open(vocab_path, 'w') as f:
        f.write(json.dumps(vocab))

    return vocab, model

# load vocabulary index from json file
def load_vocab(vocab_path='temp_embeddings/mapping.json'):
    with open(vocab_path, 'r') as f:
        data = json.loads(f.read())
    word2idx = data
    idx2word = dict([(v, k) for k, v in data.items()])
    return word2idx, idx2word


# embedding layer function
def word2vec_embedding_layer(embeddings_path='temp_embeddings/embeddings.npz'):
    weights = np.load(open(embeddings_path, 'rb'))
    layer = Embedding(input_dim=weights.shape[0], output_dim=weights.shape[1], weights=[weights])
    return layer