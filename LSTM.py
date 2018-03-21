# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
np.random.seed(42)
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "./data"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

from keras.models import Model
from keras.layers import Dense, Embedding, Input, concatenate, SpatialDropout1D
from keras.layers import LSTM, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, Dropout
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from NLP_model import clean
#embed_size = 128
embed_size = 100
max_features = 20000
maxlen = 100

file_path = "weights_base5.best.hdf5"
EMBEDDING_FILE = './data/glove.6B.100d.txt'
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")
train['comment_text'] = train['comment_text'].apply(lambda x: clean(x))
test['comment_text'] = test['comment_text'].apply(lambda x: clean(x))
train = train.sample(frac=1)

list_sentences_train = train["comment_text"].fillna("CVxTz").values
list_classes = ["toxic", "severe_toxic",
                "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("CVxTz").values

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_t = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)


def get_model():
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(LSTM(80, return_sequences=True,
                           dropout=0.1, recurrent_dropout=0.1))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    #x = GlobalMaxPool1D()(x)
    #x = Dropout(0.1)(conc)
    #x = Dense(50, activation="relu")(x)
    #x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(conc)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


embeddings_index = dict(get_coefs(*o.strip().split())
                        for o in open(EMBEDDING_FILE))
all_embs = np.stack(embeddings_index.values())
emb_mean, emb_std = all_embs.mean(), all_embs.std()

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features:
        break
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

model = get_model()
batch_size = 32
epochs = 2

checkpoint = ModelCheckpoint(
    file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early = EarlyStopping(monitor="val_loss", mode="min")

callbacks_list = [checkpoint, early]  # early
model.fit(X_t, y, batch_size=batch_size, epochs=epochs,
          validation_split=0.1, callbacks=callbacks_list)

model.load_weights(file_path)

y_test = model.predict([X_te], batch_size=1024, verbose=1)

sample_submission = pd.read_csv("./data/sample_submission.csv")
sample_submission[list_classes] = y_test
sample_submission.to_csv("./result/LSTM_baseline5.csv", index=False)
