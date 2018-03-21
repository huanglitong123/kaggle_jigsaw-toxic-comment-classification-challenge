import pandas as pd
import numpy as np
import random
random.seed(10)
#random.seed(42)
#import pdb
#train = pd.read_csv('./data/train.csv')
#X_train = train["comment_text"].fillna("fillna").values
#y_train = train[["toxic", "severe_toxic", "obscene",
#                 "threat", "insult", "identity_hate"]].values
#comment = ["toxic", "severe_toxic", "obscene",
#           "threat", "insult", "identity_hate"]
#bad_list = []
#for i in range(y_train.shape[0]):
#    result = 0
#    for j in y_train[i]:
#        result = result or j
#    if result:
#        bad_list.append(i)
#    # if i == 1000:
#    #    print bad_list
#    #    pdb.set_trace()
#print len(bad_list)
#bad_comment = pd.DataFrame()
#print bad_comment.shape
#comment_len = len(bad_list)
#for i in range(comment_len):
#    #temp = random.randint(0, 16224)
#    #bad_comment[i] = train.values[bad_list[i]] + train.values[bad_list[temp]]
#    temp = random.randint(0, 159000)
#    bad_comment[i] = train.values[bad_list[i]] + train.values[temp]
#    #pdb.set_trace()
#    if i % 10 == 0:
#        print i
#        pdb.set_trace()
#bad_comment = bad_comment.T
#print bad_comment.shape
#for i in range(2, 8):
#    for j in range(comment_len):
#        if bad_comment[i][j] > 1:
#            bad_comment[i][j] = 1
#bad_comment.columns = ["toxic", "severe_toxic", "obscene","threat", "insult", "identity_hate"]
#bad_comment.to_csv('./data/bad_comment4.csv')
bad_comment = pd.read_csv('./data/bad_comment3.csv')
train = pd.read_csv('./data/train2.csv')
train = pd.concat([train,bad_comment])
train.to_csv('./data/train2_3.csv')

