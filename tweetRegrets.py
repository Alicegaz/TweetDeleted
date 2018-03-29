from sklearn.feature_extraction.text import CountVectorizer
import re

import pandas as pd
from sklearn.cross_validation import train_test_split

import numpy as np

import string
import gensim
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import AdaBoostClassifier
from datetime import datetime
from itertools import chain
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform as sp_rand
from sklearn.model_selection import RandomizedSearchCV

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn import svm
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import operator
from collections import ChainMap
from textblob import TextBlob
from emoji import UNICODE_EMOJI
from nltk.corpus import stopwords
from imblearn.combine import SMOTEENN
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from preprocess import *



now = str(datetime.now())

def get_spam(dataset):
    spam_X = np.zeros((len(dataset), 7))
    text = dataset.iloc[:, 4]
    for index, text in enumerate(text):
        # tokens = tokens_re.findall(str(text))
        tokens = re.sub('[\s]+', ' ', text)
        tokens = text.split()
        # print(tokens)
        for token in tokens:
            token = token.strip('\'"?,.')
            if re.findall(r"#(\w+)", str(token)):
                spam_X[index, 3] += 1
            elif re.findall('@[^\s]+', str(token)):
                spam_X[index, 4] += 1
            elif re.findall(r'((www\.[^\s]+)|(https?://[^\s]+))', str(token)):
                spam_X[index, 2] += 1
            # if special_re.match(token) || ((re.match(pat, token) is None) && (token not in string.punctuation)):
            spam_X[index, 1] += 1
            spam_X[index, 0] += len(token)
            spam_X[index, 5] += sum(1 for c in token if c.isupper())
        if (spam_X[index, 0] != 0):
            spam_X[index, 5] = spam_X[index, 5] / spam_X[index, 0]
        if (spam_X[index, 1] != 0):
            spam_X[index, 6] = spam_X[index, 2] / spam_X[index, 1]
    return pd.concat([pd.DataFrame(np.reshape(np.array(spam_X), (len(spam_X), 7)),
                                   columns=['char_len', 'token_len', 'number_links', 'number_hashtags',
                                            'number_mentions', 'percent_upper', 'percent_links'])], axis=1)




def get_text_features(dataset, test_dataset, number_topics):
    print(dataset.shape)
    vectorizer = CountVectorizer(lowercase=True, tokenizer=preprocess, max_df=0.9, min_df=0.005, ngram_range=(1, 3))
    text_vector = vectorizer.fit_transform(dataset)
    corpus = gensim.matutils.Sparse2Corpus(text_vector, documents_columns=False)
    dictionary = dict([(i, s) for i, s in enumerate(vectorizer.vocabulary_.items())])
    """
    train lda model
    """
    lda = gensim.models.ldamodel.LdaModel
    ldamodel = lda(corpus, num_topics=number_topics, id2word=dictionary, passes=10)
    topics = ldamodel.print_topics(num_topics=10, num_words=10)
    topics_list = [j for _, j in topics]
    for number, topic in enumerate(topics_list):
        print('[{:d}] || {:s}\n'.format(number, topic))
    """
    assign topics to to the documents in corpus
    """
    lda_corpus = ldamodel[corpus]
    doc_features = np.zeros((len(dataset), number_topics))
    i = 0
    for x in lda_corpus:
        in_d = dict(x)
        # fill missing data, topics which are not associated with documnet with zero
        x = [(l, in_d.get(l, 0)) for l in range(number_topics)]
        doc_features[i, :] = [j for _, j in x]
        i = i + 1
    """the same for unseen"""
    doc_bow = vectorizer.transform(test_dataset)
    unseen_corpus = gensim.matutils.Sparse2Corpus(doc_bow, documents_columns=False)
    unseen_data_features = ldamodel[unseen_corpus]
    unseen_doc_features = np.zeros((len(test_dataset), number_topics))
    i = 0
    for x in unseen_data_features:
        in_d = dict(x)
        x = [(l, in_d.get(l, 0)) for l in range(number_topics)]
        unseen_doc_features[i, :] = [j for _, j in x]
        i = i + 1
        unseen_doc_features = np.array(unseen_doc_features)
    return doc_features, unseen_doc_features


def get_sentiment(s):
    analysis = TextBlob(clean_tweet(s))
    if analysis.sentiment.polarity > 0:
        return 1
    elif analysis.sentiment.polarity == 0:
        return 0
    else:
        return -1


def get_report(dataset):
    source = np.array([get_source(s) for s in dataset.iloc[:, 6]])

    unique, counts = np.unique(source, return_counts=True)
    sorted_sr = sorted(dict(zip(unique, counts)).items(), key=operator.itemgetter(1))[::-1]
    labels = [s[0] for s in sorted_sr]
    sum_source = int(sum(x[1] for x in sorted_sr))
    percentage = [int(s[1] * 100) / sum_source for s in sorted_sr]
    pie_shart = pd.Series(percentage, index=labels, name='Source')
    pie_shart.plot.pie(fontsize=11, autopct='%.2f', figsize=(6, 6));

    sent = get_sentiment_features(dataset.iloc[:, 4])
    pos_tweets = [tweet for index, tweet in enumerate(sent['SA']) if sent['SA'][index] > 0]
    neu_tweets = [tweet for index, tweet in enumerate(sent['SA']) if sent['SA'][index] == 0]
    neg_tweets = [tweet for index, tweet in enumerate(sent['SA']) if sent['SA'][index] < 0]

    print("Percentage of positive tweets: {}%".format(len(pos_tweets) * 100 / len(sent['SA'])))
    print("Percentage of neutral tweets: {}%".format(len(neu_tweets) * 100 / len(sent['SA'])))
    print("Percentage de negative tweets: {}%".format(len(neg_tweets) * 100 / len(sent['SA'])))

    """"Plot hashtags"""


def get_sentiment_features(s):
    return pd.DataFrame(
        np.reshape(np.array([get_sentiment(t) for t in s]), (len(np.array([get_sentiment(t) for t in s])))),
        columns=['SA'])


# tweeter metadata user metadata spam sentiment
def make_dataset_1(dataset):
    # hashtags = dataset.iloc[:, 10]
    tweets = dataset.iloc[:, 4]
    so = dataset.iloc[:, 6]
    source = [get_source(s) for s in so]
    spam_related = get_spam(dataset)
    source = pd.DataFrame(np.reshape(np.array(source), (len(source), 1)), columns=['source'])
    user_metadata = dataset.iloc[:, [3, 15, 16, 17, 18, 19, 31, 37]]
    user_account_data = dataset.iloc[:, [32, 33, 34, 35, 36]]
    location_metadata = dataset.iloc[:, [27]]
    time = dataset.iloc[:, 8]
    columns = []
    X = pd.concat([user_metadata, time, user_account_data, location_metadata, source, spam_related,
                   get_sentiment_features(dataset.iloc[:, 4]), tweets], axis=1)
    return X


n_folds = 5
deleted_file = '/Users/alisa/Desktop/git/tweeter/tweet-regrets/deleted.csv'
not_deleted_file = '/Users/alisa/Desktop/git/tweeter/tweet-regrets/not-deleted.csv'

dataset_deleted = pd.read_csv(deleted_file)
dataset_not_deleted = pd.read_csv(not_deleted_file)
dataset_ordered = pd.concat([dataset_not_deleted.loc[:, :], dataset_deleted.loc[:, :]])
# dataset_ordered_test = pd.concat([dataset_not_deleted.loc[1000:2000, :], dataset_deleted.loc[1001:1100, :]])
dataset = dataset_ordered.sample(frac=1).reset_index(drop=True)

dataset.replace('t', 1, inplace=True)
dataset.replace('f', 0, inplace=True)

dataset = dataset[dataset.source.notnull()]
dataset = dataset[dataset.full_text.notnull()]
dataset = dataset[dataset.profile_link_color.notnull()]
dataset = dataset[dataset.profile_sidebar_border_color.notnull()]
dataset = dataset[dataset.profile_sidebar_fill_color.notnull()]
dataset = dataset[dataset.profile_text_color.notnull()]
""" 8 - get hour from date"""
# dataset = dataset[dataset.source == ' ']

dataset = np.array(dataset)

labelencoder = LabelEncoder()
dataset[:, 32] = labelencoder.fit_transform(dataset[:, 8])

labelencoder = LabelEncoder()
dataset[:, 33] = labelencoder.fit_transform(dataset[:, 9])

labelencoder = LabelEncoder()
dataset[:, 34] = labelencoder.fit_transform(dataset[:, 10])

labelencoder = LabelEncoder()
dataset[:, 35] = labelencoder.fit_transform(dataset[:, 11])

dataset[:, 8] = [datetime.strptime(d, '%Y-%m-%d %H:%M:%S').hour for d in dataset[:, 8]]

dataset = pd.DataFrame(np.reshape(dataset, (len(dataset), len(dataset[1]))))
dataset.drop(dataset.columns[[0, 10, 18, 19, 20, 22]], axis=1)
y = np.array([0 if j == 0 else 1 for j in dataset.iloc[:, -1].fillna(0)])
X = make_dataset_1(dataset)
X = X.dropna()

# 20 19 18 22 10 0
X = np.array(X)
labelencoder = LabelEncoder()
X[:, 15] = labelencoder.fit_transform(X[:, 15])

n_folds = 10
cv = StratifiedKFold(y, n_folds=n_folds, shuffle=True)
n_f = 0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
parameters = {'kernel':('linear', 'rbf'), 'C':[1e-1, 1, 10]}
svc = SVC()
f1_scorer = make_scorer(f1_score, pos_label="yes")

X_topic, X_topic_test = get_text_features(X_train[:, -1], X_test[:, -1], 20)
X_topic = np.array(X_topic).astype('float')
X_topic_test = np.array(X_topic_test).astype('float')
np.any(np.isnan(X_topic))
np.any(np.isnan(X_topic_test))
np.all(np.isfinite(X_topic))
np.all(np.isfinite(X_topic_test))

clf = GridSearchCV(svc, parameters, n_jobs=-1, scoring=f1_scorer)
clf.fit(X_topic, y_train)
acc = clf.score(X_topic_test, y_test)
print("[INFO] grid search accuracy: {:.2f}%".format(acc * 100))
print("[INFO] grid search best parameters: {}".format(
	clf.best_params_))
