import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from NLP_model import *
import gc
import pdb
comment = ["toxic", "severe_toxic", "obscene",
           "threat", "insult", "identity_hate"]
#comment = ["toxic"]
test = pd.read_csv('./data/test.csv')
test['comment_text'] = test['comment_text'].apply(lambda x: clean(x))
y_train = test['comment_text']

for label in comment:
    path = './data/train2_'+label+'.csv'
    train = pd.read_csv(path)
    train['comment_text'] = train['comment_text'].apply(lambda x: clean(x))
    X_train = train['comment_text']
    xtrain_tfv, y_tfv = TF_data(X_train, y_train)
    #xtrain_ctv, y_ctv = Count_data(X_train, y_train)
    #xtrain_tfv_ctv = hstack([xtrain_tfv, xtrain_ctv], 'csr')
    #y_tfv_ctv = hstack([y_tfv, y_ctv], 'csr')
    xtrain_char_tfv, y_char_tfv = TF_char_data(X_train, y_train)
    xtrain_tfv_ctv = hstack([xtrain_tfv, xtrain_char_tfv], 'csr')
    y_tfv_ctv = hstack([y_tfv, y_char_tfv], 'csr')
    print label
    y = train[label]
    #model = LogisticRegression(random_state=520, class_weight="balanced")
    #model.fit(xtrain_tfv_ctv, y)
    model = NbSvmClassifier(C=1.0, dual=True).fit(xtrain_tfv_ctv, y)
    # model = LogisticRegression(
    #    C=1.0, dual=True, random_state=520, class_weight="balanced").fit(xtrain_tfv_ctv, y)
    # pdb.set_trace()
    test[label] = model.predict_proba(y_tfv_ctv)[:, 1]
    test[label].to_csv('./result/NBSVM_3_22_new_train2_' +
                       label+'.csv', index=False)
    gc.collect()

test.drop('comment_text', axis=1, inplace=True)
test.to_csv('./result/NBSVM_3_22_new_train2.csv', index=False)
#test.to_csv('./result/LR_2_21_clean2_train2_3_toxic.csv', index=False)
