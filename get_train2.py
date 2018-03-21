import pandas as pd
import numpy as np
import random
random.seed(10)
# random.seed(42)
import pdb
train = pd.read_csv('./data/train.csv')
X_train = train["comment_text"].fillna("fillna").values
y_train = train[["toxic", "severe_toxic", "obscene",
                 "threat", "insult", "identity_hate"]].values
comment = ["toxic", "severe_toxic", "obscene",
           "threat", "insult", "identity_hate"]
bad_list_all = []
k = 0
for low in range(6):
    bad_list = []
    for i in range(y_train.shape[0]):
        if y_train[i, low]:
            bad_list.append(i)
        # if i == 100:
        #    print bad_list
        #    pdb.set_trace()
    bad_comment = pd.DataFrame()
    bad_len = len(bad_list)
    comment_len = 10000
    for i in range(comment_len):
        # temp = random.randint(0, 16224)
        # bad_comment[i] = train.values[bad_list[i]] + train.values[bad_list[temp]]
        temp = random.randint(0, bad_len-1)
        temp2 = random.randint(0, bad_len-1)
        bad_comment[i] = train.values[bad_list[temp]] + \
            train.values[bad_list[temp2]]
        # if i % 100 == 0 and low == 1:
        #    print bad_comment
        #    pdb.set_trace()

    bad_comment = bad_comment.T
    print bad_comment.shape
    real = low+2
    for j in range(comment_len):
        if bad_comment[real][j] > 1:
            bad_comment[real][j] = 1
    bad_comment.columns = ["id", "comment_text", "toxic",
                           "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    path = './data/bad_comment'+comment[low]+'.csv'
    path2 = './data/train_'+comment[low]+'.csv'
    bad_comment.to_csv(path)
    bad_comment = pd.read_csv(path)
    result = pd.concat([train, bad_comment])
    result = result[["comment_text", comment[low]]]
    result.to_csv(path2)
    print low
