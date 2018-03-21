import numpy as np
import pandas as pd
from sklearn import *
from scipy.special import logit
from scipy.special import expit
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
sub1 = pd.read_csv('./result/simplest2.csv')

coly = [c for c in train.columns if c not in ['id', 'comment_text']]
y = train[coly]
tid = test['id'].values

df = pd.concat([train['comment_text'], test['comment_text']], axis=0)
df = df.fillna("unknown")
nrow = train.shape[0]

tfidf = feature_extraction.text.TfidfVectorizer(
    stop_words='english', max_features=800000)
data = tfidf.fit_transform(df)

model = ensemble.ExtraTreesClassifier(n_jobs=-1, random_state=3)
model.fit(data[:nrow], y)
print(1 - model.score(data[:nrow], y))
sub2 = model.predict_proba(data[nrow:])
sub2 = pd.DataFrame([[c[1] for c in sub2[row]] for row in range(len(sub2))]).T
sub2.columns = coly
sub2['id'] = tid
for c in coly:
    sub2[c] = sub2[c].clip(0+1e12, 1-1e12)

sub2.columns = [x+'_' if x not in ['id'] else x for x in sub2.columns]
blend = pd.merge(sub1, sub2, how='left', on='id')
for c in coly:
    blend[c] = expit(logit(blend[c] * 0.9 + blend[c+'_'] * 0.1)-0.5)
    blend[c] = blend[c].clip(0+1e12, 1-1e12)
blend = blend[sub1.columns]
blend.to_csv('./result/ExtraTrees_merge_simplest2_logit.csv', index=False)
