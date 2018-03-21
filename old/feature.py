import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
import pdb
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

# v = TfidfVectorizer(stop_words=stopwords.words("english"))
v = TfidfVectorizer()

X_train = v.fit_transform(train['comment_text'])
X_test = v.transform(test['comment_text'])
for label in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
    y = pd.DataFrame(train[label], columns=[label])

    x = pd.DataFrame(X_train.todense())
    pdb.set_trace()
    x = pd.DataFrame(X_train, columns=[i for i in range(X_train.shape[1])])
    temp = pd.concat(x, y)
    mean_label = temp[temp[label] == 1]
