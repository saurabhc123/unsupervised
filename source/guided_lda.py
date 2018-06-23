import json
import os
from pprint import pprint
from os import path
from data_processors.tweet_dataset import TweetDataSet
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from entities.tweet import Tweet
from embeddings import word2vec
from clustering import DBScan
from clustering import clusterer
import csv
import time
from experiment import Experiment
from data_processors.noise_remover import NoiseRemover
from parameters import Parameters
import re
import numpy as np
import pandas as pd
from pprint import pprint
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models.phrases import Phraser
# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
# NLTK Stop words
from nltk.corpus import stopwords
from guidedlda import guidedlda, utils

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

datafolder = 'data/classification_data/'
exports_folder = 'data/exports/'
fileName = 'Dataset_z_42_tweets.json'

# fileName = 'junk.json'
filepath = os.path.join(datafolder, fileName)

dataset = TweetDataSet(datafolder, fileName)
dataloader = DataLoader(dataset, batch_size=1)
pre_processor = NoiseRemover()

tweets = []
tweets_dict = set()
sentence_vectors = []

for data in dataloader:
    tweet = Tweet(data)
    clean_text = pre_processor.process_tweet(tweet.tweet_text)
    if clean_text not in tweets_dict:
        tweets_dict.add(clean_text)
    else:
        continue
    tweet.set_clean_text(clean_text)
    if tweet.get_word_count() > 5:
        tweets.append(tweet)
        # print(len(tweets))


def sent_to_words(tweets):
    for tweet in tweets:
        yield (gensim.utils.simple_preprocess(str(tweet.clean_text), deacc=True))  # deacc=True removes punctuations


data_words = list(sent_to_words(tweets))
print(data_words[:1])

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = Phraser(bigram)
trigram_mod = Phraser(trigram)

# See trigram example
print(trigram_mod[bigram_mod[data_words[0]]])


def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


# Form Bigrams
data_words_bigrams = make_bigrams(data_words)

data_lemmatized = data_words_bigrams

print(data_lemmatized[:1])

# Create Dictionary
vocab = corpora.Dictionary(data_lemmatized)
id2word = dict((item[1][0], item[1][1]) for item in enumerate(vocab.items()))

# Create Corpus
texts = data_lemmatized

doc_term_matrix = np.zeros((len(texts), len(vocab)), dtype=int)

# Term Document Frequency
# corpus = [id2word.doc2idx(text) for text in texts]

i = 0
for text in texts:
    indexes = vocab.doc2idx(text)
    doc_term_matrix[i][indexes] = 1
    i += 1
print("Doc-term matrix shape:", doc_term_matrix.shape)
X = doc_term_matrix
model = guidedlda.GuidedLDA(n_topics=134, n_iter=200, random_state=7, refresh=20)
model.fit(X)

topic_word = model.topic_word_
n_top_words = 8
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
    print('Topic {}: {}'.format(i, ' '.join([id2word[index] for index in topic_words])))
