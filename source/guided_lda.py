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
fileName = 'Dataset_z_1024_tweets.json'

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

words = [(bigram.vocab[key], key.decode("utf-8"))  for  key in bigram.vocab.keys()]
sorted_word_frequencies = sorted(words, key=lambda word: -(word[0]))

sorted_unigrams = [word for word in sorted_word_frequencies if "_" not in word[1]]
sorted_bigrams = [word for word in sorted_word_frequencies if "_" in word[1]]

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
word2id = dict((item[1][1], item[1][0]) for item in enumerate(vocab.items()))

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

seed_topic_list = [['help', 'victim','newtown', 'family'],
                   ['mass', 'newtown'],
                   ['tribute','vigil','picket','baptist','church'],
                   ['victim', 'tragedy', 'mourn', 'rip','innocent','sad','child'],
                   ['fund', 'donate', 'pour', 'raise'],
                   ['family', 'prayer', 'heart', 'silence','lose','child','parent'],
                   ['month', 'anniversary', 'commemorate', 'remembrance', 'since', 'year'],
                   ['nra', 'gun', 'control', 'arm', 'stop', 'good', 'bad', 'guard'],
                   ['hold', 'funeral'],
                   ['obama', 'president', 'speech', 'speak', 'barrack'],
                   ['kill', 'massacre', 'die', 'child','dead','death','include'],
                   ['survivor', 'sue'],
                   ['fake','shit']
                   ]

seed_topics = {}
for t_id, st in enumerate(seed_topic_list):
    for word in st:
        try:
            seed_topics[word2id[word]] = t_id
        except:
            print("Word {} not found in dictionary for seeding topic {}.".format(word,t_id))
            pass

seed_confidence=0.15
n_topics=134
n_iter=200
model = guidedlda.GuidedLDA(n_topics=n_topics, n_iter=n_iter, random_state=7, refresh=20)
model.fit(X,seed_topics=seed_topics, seed_confidence=seed_confidence)

topic_word = model.topic_word_
n_top_words = 8
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
    print('Topic {}: {}'.format(i, ' '.join([id2word[index] for index in topic_words])))

doc_topics = model.transform(X)

exports_folder = 'data/exports/'
timestamp = time.strftime("%Y%m%d-%H%M%S")
exports_filename = 'guided_LDA_'+ str(seed_confidence) + "_" + fileName + "_"+ timestamp + '.csv'
exports_filepath = os.path.join(exports_folder,exports_filename)
with open(exports_filepath,'w') as out:
    csv_out=csv.writer(out, delimiter = '|')
    csv_out.writerow(['label' ,'tweet_text'])
    for i in range(len(X)):
        info = (doc_topics[i].argmax(), X[i])
        tweet = tweets[i]
        csv_out.writerow([doc_topics[i].argmax(), tweet.tweet_text, tweet.clean_text])

parameters = Parameters()
parameters.add_parameter("num_topics", n_topics)
parameters.add_parameter("num_iterations", n_iter)
parameters.add_parameter("seed_probability", seed_confidence)
parameters.add_complex_parameter("seed_topics", seed_topic_list)
parameters.add_complex_parameter("unigrams_counts", sorted_unigrams)
parameters.add_complex_parameter("bigrams_counts", sorted_bigrams)
parameters.write_parameters(exports_folder, timestamp)

#Generate similarity for all sub-clusters


