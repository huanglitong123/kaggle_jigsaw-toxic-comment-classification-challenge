import numpy as np
np.random.seed(42)
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate, Dropout
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from NLP_model import clean
from attlayer import AttentionWeightedAverage, Attention, AttentionWithContext
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['OMP_NUM_THREADS'] = '4'


EMBEDDING_FILE = './data/glove.6B.100d.txt'

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
#train['comment_text'] = train['comment_text'].apply(lambda x: clean(x))
#test['comment_text'] = test['comment_text'].apply(lambda x: clean(x))
submission = pd.read_csv('./data/sample_submission.csv')

X_train = train["comment_text"].fillna("fillna").values
y_train = train[["toxic", "severe_toxic", "obscene",
                 "threat", "insult", "identity_hate"]].values
X_test = test["comment_text"].fillna("fillna").values


max_features = 20000
maxlen = 100
embed_size = 100

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train) + list(X_test))
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
x_test = sequence.pad_sequences(X_test, maxlen=maxlen)


def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')


embeddings_index = dict(get_coefs(*o.strip().split())
                        for o in open(EMBEDDING_FILE))

all_embs = np.stack(embeddings_index.values())
emb_mean, emb_std = all_embs.mean(), all_embs.std()

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))


def get_model():
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(80, return_sequences=True))(x)
    #x = Dropout(0.5)(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    #awc = AttentionWithContext()(x)
    #att = Attention()(x)
    #atten = AttentionWeightedAverage()(x)
    conc = concatenate([avg_pool, max_pool])
    #conc = concatenate([awc, att])
    #x = Dense(50, activation="relu")(awc)
    #x = Dropout(0.5)(x)
    outp = Dense(6, activation="sigmoid")(conc)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


model = get_model()

file_path = "./weight/weights_base_GRU_awc2_test.best.hdf5"
batch_size = 32
epochs = 10

[X_tra, X_val, y_tra, y_val] = train_test_split(
    x_train, y_train, train_size=0.95, random_state=233)
RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)
early = EarlyStopping(monitor="val_loss", mode="min")
checkpoint = ModelCheckpoint(
    file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')


def exp_decay(init, fin, steps): return (init/fin)**(1/(steps-1)) - 1


steps = int(len(X_tra)/batch_size) * epochs
lr_init, lr_fin = 0.001, 0.0005
lr_decay = exp_decay(lr_init, lr_fin, steps)
K.set_value(model.optimizer.lr, lr_init)
K.set_value(model.optimizer.decay, lr_decay)

hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),
                 callbacks=[RocAuc, early, checkpoint], verbose=1)


y_pred = model.predict(x_test, batch_size=1024)
submission[["toxic", "severe_toxic", "obscene",
            "threat", "insult", "identity_hate"]] = y_pred
submission.to_csv('./result/GRU_awc2_test_submission.csv', index=False)
