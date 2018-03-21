
# coding: utf-8

# # Approaching (Almost) Any NLP Problem on Kaggle
#
# In this post I'll talk about approaching natural language processing problems on Kaggle. As an example, we will use the data from this competition. We will create a very basic first model first and then improve it using different other features. We will also see how deep neural networks can be used and end this post with some ideas about ensembling in general.
#
# ### This covers:
# - tfidf
# - count features
# - logistic regression
# - naive bayes
# - svm
# - xgboost
# - grid search
# - word vectors
# - LSTM
# - GRU
# - Ensembling
#
# *NOTE*: This notebook is not meant for achieving a very high score on the Leaderboard for this dataset. However, if you follow it properly, you can get a very high score with some tuning. ;)
#
# So, without wasting any time, let's start with importing some important
# python modules that I'll be using.
import os
mingw_path = 'C:\Program Files\mingw-w64\\x86_64-7.2.0-posix-seh-rt_v5-rev1\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
import re
from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np
import xgboost as xgb
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_curve, auc
from nltk import word_tokenize
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix, hstack
from NBSVM import NbSvmClassifier
from nltk.tokenize import TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from appos import APPO
stop_words = stopwords.words('english')
eng_stopwords = set(stopwords.words("english"))

# The problem requires us to predict the author, i.e. EAP, HPL and MWS given the text. In simpler words, text classification with 3 different classes.
#
# For this particular problem, Kaggle has specified multi-class log-loss
# as evaluation metric. This is implemented in the follow way (taken from:
# https://github.com/dnouri/nolearn/blob/master/nolearn/lasagne/util.py)


def multiclass_logloss(actual, predicted, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    :param actual: Array containing the actual target classes
    :param predicted: Matrix with class predictions, one probability per class
    """
    # Convert 'actual' to a binary array if it's not already:
    #print(type(actual), actual.shape, type(predicted), predicted.shape)
    frp, trp, thres = roc_curve(actual, predicted[:, 1])
    auc_val = auc(frp, trp)
    print(['auc is ', auc_val])
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota


# ## Building Basic Models
# Our very first model is a simple TF-IDF (Term Frequency - Inverse Document Frequency) followed by a simple Logistic Regression.
# Always start with these features. They work (almost) everytime!
def TF_data(xtrain, xvalid):
    tfv = TfidfVectorizer(min_df=3,  max_features=None,
                          strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                          ngram_range=(1, 2), use_idf=1, smooth_idf=1, sublinear_tf=1,
                          stop_words='english')

    # Fitting TF-IDF to both training and test sets (semi-supervised learning)
    tfv.fit(list(xtrain))
    #tfv.fit(list(xtrain) + list(xvalid))
    xtrain_tfv = tfv.transform(xtrain)
    xvalid_tfv = tfv.transform(xvalid)
    return xtrain_tfv, xvalid_tfv


def TF_char_data(xtrain, xvalid):
    tfv = TfidfVectorizer(min_df=3,  max_features=None,
                          strip_accents='unicode', analyzer='char', token_pattern=r'\w{1,}',
                          ngram_range=(1, 4), use_idf=1, smooth_idf=1, sublinear_tf=1,
                          stop_words='english')

    # Fitting TF-IDF to both training and test sets (semi-supervised learning)
    tfv.fit(list(xtrain))
    #tfv.fit(list(xtrain) + list(xvalid))
    xtrain_tfv = tfv.transform(xtrain)
    xvalid_tfv = tfv.transform(xvalid)
    return xtrain_tfv, xvalid_tfv
# Fitting a simple Logistic Regression on TFIDF


def LR_model(xtrain_tfv, ytrain, xvalid_tfv, yvalid):
    clf = LogisticRegression(
        C=1.0, dual=True, random_state=520, class_weight="balanced")
    #clf = LogisticRegression(C=1.0, random_state=520)
    clf.fit(xtrain_tfv, ytrain)
    #predictions = clf.predict_proba(xvalid_tfv)
    predictions = clf.predict_proba(xvalid_tfv)

    print("LR_model logloss: %0.3f " % multiclass_logloss(yvalid, predictions))
    return clf


# And there we go. We have our first model with a multiclass logloss of 0.626.
#
# But we are greedy and want a better score. Lets look at the same model with a different data.
#
# Instead of using TF-IDF, we can also use word counts as features. This
# can be done easily using CountVectorizer from scikit-learn.


def Count_data(xtrain, xvalid):
    ctv = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}',
                          ngram_range=(1, 3), stop_words='english')

    # Fitting Count Vectorizer to both training and test sets (semi-supervised
    # learning)
    ctv.fit(list(xtrain))
    #ctv.fit(list(xtrain) + list(xvalid))
    xtrain_ctv = ctv.transform(xtrain)
    xvalid_ctv = ctv.transform(xvalid)
    return xtrain_ctv, xvalid_ctv

# Aaaaanddddddd Wallah! We just improved our first model by 0.1!!!
#
# Next, let's try a very simple model which was quite famous in ancient times - Naive Bayes.
# Let's see what happens when we use naive bayes on these two datasets:
# Fitting a simple Naive Bayes on TFIDF


def NB_model(xtrain_tfv, ytrain, xvalid_tfv, yvalid):
    clf = MultinomialNB()
    clf.fit(xtrain_tfv, ytrain)
    predictions = clf.predict_proba(xvalid_tfv)

    print("NB_model logloss: %0.3f " % multiclass_logloss(yvalid, predictions))
    return clf

# Good performance! But the logistic regression on counts is still better!
# What happens when we use this model on counts data instead?

# Fitting a simple Naive Bayes on Counts
# Whoa! Seems like old stuff still works good!!!! One more ancient algorithms in the list is SVMs. Some people "love" SVMs. So, we must try SVM on this dataset.
#
# Since SVMs take a lot of time, we will reduce the number of features from the TF-IDF using Singular Value Decomposition before applying SVM.
#
# Also, note that before applying SVMs, we *must* standardize the data.
# Apply SVD, I chose 120 components. 120-200 components are good enough
# for SVM model.


def SVD_data(xtrain_tfv, xvalid_tfv):
    svd = decomposition.TruncatedSVD(n_components=120)
    svd.fit(xtrain_tfv)
    xtrain_svd = svd.transform(xtrain_tfv)
    xvalid_svd = svd.transform(xvalid_tfv)

    # Scale the data obtained from SVD. Renaming variable to reuse without
    # scaling.
    scl = preprocessing.StandardScaler()
    scl.fit(xtrain_svd)
    xtrain_svd_scl = scl.transform(xtrain_svd)
    xvalid_svd_scl = scl.transform(xvalid_svd)
    xtrain_svd = csr_matrix(xtrain_svd)
    xvalid_svd = csr_matrix(xvalid_svd)
    return xtrain_svd, xvalid_svd, xtrain_svd_scl, xvalid_svd_scl


# Now it's time to apply SVM. After running the following cell, feel free
# to go for a walk or talk to your girlfriend/boyfriend. :P

# Fitting a simple SVM


def SVC_model(xtrain_svd_scl, ytrain, xvalid_svd_scl, yvalid):
    # since we need probabilities
    clf = SVC(C=1.0, probability=True, random_state=520)
    clf.fit(xtrain_svd_scl, ytrain)
    predictions = clf.predict_proba(xvalid_svd_scl)

    print("SVC_model logloss: %0.3f " %
          multiclass_logloss(yvalid, predictions))
    return clf


def SVC_model_2(xtrain_svd_scl, ytrain, xvalid_svd_scl, yvalid):
    # since we need probabilities
    clf = LinearSVC(random_state=520)
    clf.fit(xtrain_svd_scl, ytrain)
    predictions = clf.predict(xvalid_svd_scl)
    frp, trp, thres = roc_curve(yvalid, predictions)
    auc_val = auc(frp, trp)
    print(['auc is ', auc_val])
    # print("SVC_model logloss: %0.3f " %
    #      multiclass_logloss(yvalid, predictions))
    return clf

# Oops! time to get up! Looks like SVM doesn't perform well on this data...!
# Before moving further, lets apply the most popular algorithm on Kaggle:
# xgboost!
# Fitting a simple xgboost on tf-idf
# Fitting a simple xgboost on counts
# Fitting a simple xgboost on tf-idf svd features


def XGB_model(xtrain_tfv, ytrain, xvalid_tfv, yvalid):
    clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8,
                            subsample=0.8, nthread=10, learning_rate=0.1, random_state=520)
    clf.fit(xtrain_tfv.tocsc(), ytrain, eval_metric='auc')
    predictions = clf.predict_proba(xvalid_tfv.tocsc())

    print("XGB_model logloss: %0.3f " %
          multiclass_logloss(yvalid, predictions))
    return clf


# Fitting a simple xgboost on tf-idf svd features
def XGB_model1(xtrain_svd, ytrain, xvalid_svd, yvalid, silent=True):
    clf = xgb.XGBClassifier(nthread=10, silent=silent, random_state=520)
    clf.fit(xtrain_svd, ytrain, eval_metric='auc')
    predictions = clf.predict_proba(xvalid_svd)

    print("XGB_model1 logloss: %0.3f " %
          multiclass_logloss(yvalid, predictions))
    return clf
# Seems like no luck with XGBoost! But that is not correct. I haven't done
# any hyperparameter optimizations yet. And since I'm lazy, I'll just tell
# you how to do it and you can do it on your own! ;). This will be
# discussed in the next section:


def NBSVM_model(xtrain_svd, ytrain, xvalid_svd, yvalid, silent=True):
    clf = NbSvmClassifier(C=1, dual=True).fit(xtrain_svd, ytrain)
    predictions = clf.predict_proba(xvalid_svd)

    print("NBSVM_model logloss: %0.3f " %
          multiclass_logloss(yvalid, predictions))
    return clf


def Word_Vectors_xgb(xtrain, ytrain, xvalid, yvalid):
    # This is an improvement of 8% over the original naive bayes score!
    #
    # In NLP problems, it's customary to look at word vectors. Word vectors give a lot of insights about the data. Let's dive into that.
    #
    # ## Word Vectors
    #
    # Without going into too much details, I would explain how to create
    # sentence vectors and how can we use them to create a machine learning
    # model on top of it. I am a fan of GloVe vectors, word2vec and fasttext.
    # In this post, I'll be using the GloVe vectors. You can download the
    # GloVe vectors from here
    # `http://www-nlp.stanford.edu/data/glove.840B.300d.zip`
    # load the GloVe vectors in a dictionary:

    embeddings_index = {}
    f = open('glove.840B.300d.txt')
    for line in tqdm(f):
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    # this function creates a normalized vector for the whole sentence
    def sent2vec(s):
        words = str(s).lower().decode('utf-8')
        words = word_tokenize(words)
        words = [w for w in words if not w in stop_words]
        words = [w for w in words if w.isalpha()]
        M = []
        for w in words:
            try:
                M.append(embeddings_index[w])
            except:
                continue
        M = np.array(M)
        v = M.sum(axis=0)
        if type(v) != np.ndarray:
            return np.zeros(300)
        return v / np.sqrt((v ** 2).sum())

    # create sentence vectors using the above function for training and
    # validation set
    xtrain_glove = [sent2vec(x) for x in tqdm(xtrain)]
    xvalid_glove = [sent2vec(x) for x in tqdm(xvalid)]
    xtrain_glove = np.array(xtrain_glove)
    xvalid_glove = np.array(xvalid_glove)

    # Let's see the performance of xgboost on glove features:
    # Fitting a simple xgboost on glove features
    clf = XGB_model1(xtrain_svd, ytrain, xvalid_svd, yvalid, False)
    # Fitting a simple xgboost on glove features
    clf = XGB_model(xtrain_glove, ytrain, xvalid_glove, yvalid)


def clean(comment):
    """
    This function receives comments and returns clean word-list
    """
    # Convert to lower case , so that Hi and hi are the same
    comment = comment.lower()
    # remove \n
    comment = re.sub("\\n", "", comment)
    # remove leaky elements like ip,user
    comment = re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", "spec_ip", comment)
    # removing usernames
    comment = re.sub("\[\[.*\]", "spec_usernames", comment)

    # Split the sentences into words
    words = TweetTokenizer().tokenize(comment)

    # (')aphostophe  replacement (ie)   you're --> you are
    # ( basic dictionary lookup : master dictionary present in a hidden block of code)
    words = [APPO[word] if word in APPO else word for word in words]
    words = [WordNetLemmatizer().lemmatize(word, "v") for word in words]
    words = [w for w in words if not w in eng_stopwords]

    clean_sent = " ".join(words)
    # remove any non alphanum,digit character
    #clean_sent=re.sub("\W+"," ",clean_sent)
    #clean_sent=re.sub("  "," ",clean_sent)
    return(clean_sent)


def main():
    train = pd.read_csv('./data/train.csv')
    test = pd.read_csv('./data/test.csv')
    sample = pd.read_csv('./data/sample_submission.csv')

    train["comment_text"].fillna("unknown", inplace=True)
    test["comment_text"].fillna("unknown", inplace=True)
    X_train = train['comment_text'].apply(lambda x: clean(x))
    y_train = test['comment_text'].apply(lambda x: clean(x))

    # We use the LabelEncoder from scikit-learn to convert text labels to
    # integers, 0, 1 2
    for label in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
        print('start ', label, ' pred ')
        y = train[label]
        # Before going further it is important that we split the data into
        # training and validation sets. We can do it using `train_test_split` from
        # the `model_selection` module of scikit-learn.
        xtrain, xvalid, ytrain, yvalid = train_test_split(X_train, y,
                                                          stratify=y,
                                                          random_state=42,
                                                          test_size=0.2, shuffle=True)
        print(xtrain.shape)
        print(xvalid.shape)
        xtrain_tfv, xvalid_tfv = TF_data(xtrain, xvalid)
        #clf = LR_model(xtrain_tfv, ytrain, xvalid_tfv, yvalid)
        clf = NBSVM_model(xtrain_tfv, ytrain, xvalid_tfv, yvalid)
        xtrain_char, xvalid_char = TF_char_data(xtrain, xvalid)
        xtrain_char_tfv = hstack([xtrain_tfv, xtrain_char], 'csr')
        xvalid_char_tfv = hstack([xvalid_tfv, xvalid_char], 'csr')

        clf = NBSVM_model(xtrain_char_tfv, ytrain, xvalid_char_tfv, yvalid)

        clf = LR_model(xtrain_char, ytrain, xvalid_char, yvalid)

        clf = LR_model(xtrain_char_tfv, ytrain, xvalid_char_tfv, yvalid)
        #clf = NB_model(xtrain_tfv, ytrain, xvalid_tfv, yvalid)
        #clf = XGB_model(xtrain_tfv, ytrain, xvalid_tfv, yvalid)

        #xtrain_ctv, xvalid_ctv = Count_data(xtrain, xvalid)
        #xtrain_tfv_ctv = hstack([xtrain_tfv, xtrain_ctv], 'csr')
        #xvalid_tfv_ctv = hstack([xvalid_tfv, xvalid_ctv], 'csr')
        #clf = LR_model(xtrain_ctv, ytrain, xvalid_ctv, yvalid)
        #clf = LR_model(xtrain_tfv_ctv, ytrain, xvalid_tfv_ctv, yvalid)
        #clf = NB_model(xtrain_ctv, ytrain, xvalid_ctv, yvalid)
        #clf = XGB_model(xtrain_ctv, ytrain, xvalid_ctv, yvalid)
        #clf = XGB_model(xtrain_tfv_ctv, ytrain, xvalid_tfv_ctv, yvalid)

        xtrain_svd, xvalid_svd, xtrain_svd_scl, xvalid_svd_scl = SVD_data(
            xtrain_tfv, xvalid_tfv)
        #clf = LR_model(xtrain_svd_scl, ytrain, xvalid_svd_scl, yvalid)
        #clf = LR_model(xtrain_svd, ytrain, xvalid_svd, yvalid)
        clf = SVC_model_2(xtrain_svd_scl, ytrain, xvalid_svd_scl, yvalid)
        clf = SVC_model(xtrain_svd_scl, ytrain, xvalid_svd_scl, yvalid)

        #clf = XGB_model(xtrain_svd, ytrain, xvalid_svd, yvalid)
        #clf = XGB_model1(xtrain_svd, ytrain, xvalid_svd, yvalid)
if __name__ == "__main__":
    main()
    # high_make_train_set()
    # get_weight()
