#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
import os
mingw_path = 'C:\Program Files\mingw-w64\\x86_64-7.2.0-posix-seh-rt_v5-rev1\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import xgboost as xgb
import gc
from sklearn.feature_extraction.text import TfidfVectorizer

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

train = train.fillna("unknown")
test = test.fillna("unknown")


train_mes, valid_mes, train_l, valid_l = train_test_split(train['comment_text'], train[
                                                          ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']], test_size=0.2, random_state=2)

# Using the tokenize function from Jeremy's kernel
import re
import string
re_tok = re.compile(u'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')


def tokenize(s): return re_tok.sub(r' \1 ', s).split()

v = TfidfVectorizer(ngram_range=(1, 2), tokenizer=tokenize,
                    min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
                    smooth_idf=1, sublinear_tf=1)
'''comments_train = transform_com.transform(train['comment_text'])'''
transform_com = v.fit_transform(train['comment_text'])
comments_train = v.transform(train_mes)
comments_valid = v.transform(valid_mes)
comments_test = v.transform(test['comment_text'])
gc.collect()


import xgboost as xgb


def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=2017, num_rounds=400):
    param = {}
    param['objective'] = 'binary:logistic'
    param['eta'] = 0.12
    param['max_depth'] = 5
    param['silent'] = 1
    param['eval_metric'] = 'auc'
    param['min_child_weight'] = 1
    param['subsample'] = 0.5
    param['colsample_bytree'] = 0.7
    param['seed'] = seed_val
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)
    xgtest = xgb.DMatrix(test_X, label=test_y)
    watchlist = [(xgtrain, 'train'), (xgtest, 'test')]
    model = xgb.train(plst, xgtrain, num_rounds,
                      watchlist, early_stopping_rounds=20)
    return model


col = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
preds = np.zeros((test.shape[0], len(col)))

for i, j in enumerate(col):
    print('fit '+j)
    model = runXGB(comments_train, train_l[j], comments_valid, valid_l[j])
    preds[:, i] = model.predict(xgb.DMatrix(comments_test))
    gc.collect()

subm = pd.read_csv('./data/sample_submission.csv')
submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns=col)], axis=1)
submission.to_csv('./result/test_xgb3.csv', index=False)


# Any results you write to the current directory are saved as output.
