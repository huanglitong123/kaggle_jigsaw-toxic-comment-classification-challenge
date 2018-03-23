import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from NLP_model import *
import gc
import pdb
train = pd.read_csv('./data/train_RNN2.csv')
test = pd.read_csv('./data/test.csv')
#xtrain_count, y_count = count_data(train, test)
train['comment_text'] = train['comment_text'].apply(lambda x: clean(x))
test['comment_text'] = test['comment_text'].apply(lambda x: clean(x))
X_train = train['comment_text']
y_train = test['comment_text']
xtrain_tfv, y_tfv = TF_data(X_train, y_train)
#xtrain_ctv, y_ctv = Count_data(X_train, y_train)
#xtrain_tfv_ctv = hstack([xtrain_tfv, xtrain_ctv], 'csr')
#y_tfv_ctv = hstack([y_tfv, y_ctv], 'csr')
xtrain_char_tfv, y_char_tfv = TF_char_data(X_train, y_train)
xtrain_tfv_ctv = hstack([xtrain_tfv, xtrain_char_tfv], 'csr')
y_tfv_ctv = hstack([y_tfv, y_char_tfv], 'csr')

for label in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
    # for label in ['toxic']:
    print label
    y = train[label]
    #model = LogisticRegression(random_state=520, class_weight="balanced")
    #model.fit(xtrain_tfv_ctv, y)
    model = NbSvmClassifier(C=1.0, dual=True).fit(xtrain_tfv_ctv, y)
    # model = LogisticRegression(
    #    C=1.0, dual=True, random_state=520, class_weight="balanced").fit(xtrain_tfv_ctv, y)
    # pdb.set_trace()
    test[label] = model.predict_proba(y_tfv_ctv)[:, 1]
    test[label].to_csv('./result/NBSVM_3_21_new_train_' +
                       label+'.csv', index=False)
    gc.collect()

test.drop('comment_text', axis=1, inplace=True)
test.to_csv('./result/NBSVM_3_21_train_RNN.csv', index=False)
#test.to_csv('./result/LR_2_21_clean2_train2_3_toxic.csv', index=False)
