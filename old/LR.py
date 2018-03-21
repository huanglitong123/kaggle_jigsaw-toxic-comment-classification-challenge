import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from NLP_model import *
import gc
import pdb
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

#v = TfidfVectorizer(stop_words=stopwords.words("english"))#
#X_train = v.fit_transform(train['comment_text'])
#X_test = v.transform(test['comment_text'])
#X_train = train['comment_text']
#y_train = test['comment_text']
X_train = train['comment_text'].apply(lambda x: clean(x))
y_train = test['comment_text'].apply(lambda x: clean(x))
xtrain_tfv, y_tfv = TF_data(X_train, y_train)
#xtrain_ctv, y_ctv = Count_data(X_train, y_train)
#xtrain_tfv_ctv = hstack([xtrain_tfv, xtrain_ctv], 'csr')
#y_tfv_ctv = hstack([y_tfv, y_ctv], 'csr')
xtrain_char_tfv, y_char_tfv = TF_char_data(X_train, y_train)
xtrain_tfv_ctv = hstack([xtrain_tfv, xtrain_char_tfv], 'csr')
y_tfv_ctv = hstack([y_tfv, y_char_tfv], 'csr')

for label in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
    y = train[label]
    #model = LogisticRegression(random_state=520, class_weight="balanced")
    #model.fit(xtrain_tfv_ctv, y)
    model = NbSvmClassifier(C=1.0, dual=True, n_jobs=-1).fit(xtrain_tfv_ctv, y)
    # pdb.set_trace()
    test[label] = model.predict_proba(y_tfv_ctv)[:, 1]
    gc.collect()

test.drop('comment_text', axis=1, inplace=True)
test.to_csv('./result/NBSVM_tfv_char(1-4)2_clean2.csv', index=False)


def add_feature(X, feature_to_add):
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')
