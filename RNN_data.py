import pandas as pd
import numpy as np
import random
from sklearn import preprocessing
# random.seed(10)
random.seed(42)
import pdb

train = pd.read_csv('./data/train.csv')
X_train = train["comment_text"].fillna("fillna").values
y_train = train[["toxic", "severe_toxic", "obscene",
                 "threat", "insult", "identity_hate"]].values
comment = ["toxic", "severe_toxic", "obscene",
           "threat", "insult", "identity_hate"]
bad_class = []
print y_train.shape[0]
for i in range(y_train.shape[0]):
    temp = 0
    for j in range(6):
        temp = temp + y_train[i, j]*(10**j)
    bad_class.append(temp)
le = preprocessing.LabelEncoder()
le.fit(bad_class)
print le.classes_
for i in le.classes_:
    bad_list = [index for index, x in enumerate(bad_class) if x == i]
    # pdb.set_trace()
    bad_comment = pd.DataFrame()
    bad_len = len(bad_list)
    if bad_len > 3000:
        comment_len = 2000
    elif bad_len > 1000:
        comment_len = bad_len*2
    elif bad_len > 500:
        comment_len = bad_len*4
    elif bad_len < 500:
        comment_len = bad_len*10
    print(i, comment_len)
    for i1 in range(comment_len):
        temp = random.randint(0, bad_len-1)
        temp2 = random.randint(0, bad_len-1)
        bad_comment[i1] = train.values[bad_list[temp]] + \
            train.values[bad_list[temp2]]
    bad_comment = bad_comment.T
    print bad_comment.shape
    for i2 in range(2, 8):
        for j in range(comment_len):
            if bad_comment[i2][j] > 1:
                bad_comment[i2][j] = 1
    bad_comment.columns = ["id", "comment_text", "toxic",
                           "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    path = './RNN_data/bad_comment_RNN2_'+str(i)+'.csv'
    bad_comment.to_csv(path)
    # pdb.set_trace()

path2 = './data/train_RNN2.csv'
result = train
for i in le.classes_:
    path = './RNN_data/bad_comment_RNN2_'+str(i)+'.csv'
    bad_comment = pd.read_csv(path)
    result = pd.concat([result, bad_comment], join='inner', ignore_index=True)
    # pdb.set_trace()
# pdb.set_trace()
result.to_csv(path2)
