import numpy as np
from keras.preprocessing import sequence
from keras.models import Model, Sequential, model_from_json
from keras.layers.wrappers import Bidirectional
from keras.layers import Activation, Input, TimeDistributed, LSTM, Dense, Dropout, Embedding
from keras.callbacks import EarlyStopping
from keras.models import save_model, load_model
from keras_contrib.layers import CRF
from keras_contrib.utils import save_load_utils

from gensim.models import Word2Vec
from embedding import load_vocab

# params
MAX_LENGTH = 25      # 2x avg  sentence length
MAX_VOCAB = 18000    # see preprocessing.ipynb
EMBEDDING_SIZE = 128 # specified w2v size, see preprocessing.ipynb
HIDDEN_SIZE = 256    # LSTM Nodes/Features/Dimension
BATCH_SIZE = 128
DROPOUTRATE = 0.0
MAX_EPOCHS = 10 # max iterations, early stop condition below

# load data
print("loading data...\n")
vocab = list(np.load('data/vocab.npy'))
word2idx = np.load('data/word2idx.npy').item()
idx2word = np.load('data/idx2word.npy').item()
X_train = np.load('data/X_train.npy')
X_test = np.load('data/X_test.npy')
y_train = np.load('data/y_train.npy')
y_test = np.load('data/y_test.npy')
# load embedding data
w2v_vocab, _ = load_vocab('embeddings/text_mapping.json')
w2v_model = Word2Vec.load('embeddings/text_embeddings.gensimmodel')

# pad data
print("zero-padding sequences...\n")
X_train = sequence.pad_sequences(X_train, maxlen=MAX_LENGTH, truncating='post', padding='post')
X_test = sequence.pad_sequences(X_test, maxlen=MAX_LENGTH, truncating='post', padding='post')
y_train = sequence.pad_sequences(y_train, maxlen=MAX_LENGTH, truncating='post', padding='post')
y_test = sequence.pad_sequences(y_test, maxlen=MAX_LENGTH, truncating='post', padding='post')
# reshape data for CRF
y_train = y_train[:, :, np.newaxis]
y_test = y_test[:, :, np.newaxis]
print(y_train[:3])

# embedding matrices
print("creating embedding matrices...\n")

word_embedding_matrix = np.zeros((MAX_VOCAB, EMBEDDING_SIZE))

for word in word2idx.keys():
    # get the word vector from the embedding model
    # if it's there (check against vocab list)
    if word in w2v_vocab:
        # get the word vector
        word_vector = w2v_model[word]
        # slot it in at the proper index
word_embedding_matrix[word2idx[word]] = word_vector

# model
print('Building model...\n')
# text layers : dense embedding > dropout > bi-LSTM
txt_input = Input(shape=(MAX_LENGTH,), name='txt_input')
txt_embed = Embedding(MAX_VOCAB, EMBEDDING_SIZE, input_length=MAX_LENGTH,
                      weights=[word_embedding_matrix],
                      trainable=True,
                      mask_zero=True,
                      name='txt_embedding')(txt_input)
txt_drpot = Dropout(DROPOUTRATE, name='txt_dropout')(txt_embed)
txt_lstml = Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True),
                          name='txt_bidirectional_1')(txt_drpot)
txt_lstml = Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True),
                          name='txt_bidirectional_2')(txt_lstml)
txt_lstml = Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True),
                          name='txt_bidirectional_3')(txt_lstml)

# final linear chain CRF layer
crf = CRF(2, sparse_target=True)
mrg_chain = crf(txt_lstml)

model = Model(inputs=[txt_input], outputs=mrg_chain)

earlystop = EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,
    patience=1,
    mode=min
)

callbacks_list = [earlystop]

model.compile(optimizer='adam',
              loss=crf.loss_function,
              metrics=[crf.accuracy])

model.summary()

history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    batch_size=BATCH_SIZE,
                    callbacks=callbacks_list,
                    epochs=MAX_EPOCHS)

hist_dict = history.history

# model.save('model/crf_model.h5')
save_load_utils.save_all_weights(model, 'model/crf_model.h5')
np.save('model/hist_dict.npy', hist_dict)
print("models saved!\n")

scores = model.evaluate(X_test, y_test)
print('')
print('Eval model...')
print("Accuracy: %.2f%%" % (scores[1] * 100), '\n')

# CRF: https://github.com/farizrahman4u/keras-contrib/blob/master/keras_contrib/layers/crf.py
# loading keras-contrib: https://github.com/farizrahman4u/keras-contrib/issues/125
