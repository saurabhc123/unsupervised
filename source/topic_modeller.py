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
from data_processors.noise_remover import  NoiseRemover
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
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

datafolder = 'data/classification_data/'
exports_folder = 'data/exports/'
fileName = 'Dataset_z_41_tweets.json'

#fileName = 'junk.json'
filepath = os.path.join(datafolder,fileName)


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
    #print(len(tweets))


def sent_to_words(tweets):
    for tweet in tweets:
        yield(gensim.utils.simple_preprocess(str(tweet.clean_text), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(tweets))
print(data_words[:1])
# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
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
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus[:3])

# Human readable format of corpus (term-frequency)
print ([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:3]])


#Build LDA model
def get_generic_lda_model(corpus, id2word, num_topics=20):
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=num_topics,
                                               random_state=100,
                                               update_every=1,
                                               chunksize=100,
                                               passes=10,
                                               alpha='auto',
                                               eta='auto',
                                               per_word_topics=True)

    # Print the Keyword in the 10 topics
    pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]
    return lda_model

def get_mallet_lda_model(corpus, id2word, num_topics=20):
    mallet_path = 'source/mallet-2.0.8/bin/mallet' # update this path
    ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
    return ldamallet

def show_model_statistics(lda_model, with_visualization=False):
    # Compute Perplexity
    #print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.
    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)
    # Visualize the topics
    # pyLDAvis.enable_notebook()
    if with_visualization:
        vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
        pyLDAvis.show(vis)




def compute_coherence_values(dictionary, corpus, texts, limit, model, start=2, step=3):
    """
    Compute c_v coherence for various number of topics
    best_coherence = 0.0
    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    best_coherence = 0.0
    best_num_of_topics = 2
    best_model = 0
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = get_mallet_lda_model(corpus, id2word, num_topics=num_topics)
        #model.num_topics = num_topics
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence = coherencemodel.get_coherence()
        if coherence > best_coherence:
            best_coherence = coherence
            best_num_of_topics = num_topics
            best_model = model
        coherence_values.append(coherence)

    return model_list, coherence_values, best_model, best_num_of_topics



model = get_mallet_lda_model(corpus, id2word, num_topics=90)

#model = get_generic_lda_model(corpus, id2word)

#show_model_statistics(model, True)

def find_best_num_of_topics(corpus, id2word, data_lemmatized, model):
    # Can take a long time to run.
    start=2
    limit=150
    step=6
    model_list, coherence_values, best_model, best_num_of_topics = compute_coherence_values(dictionary=id2word, corpus=corpus,
                                                            texts=data_lemmatized, model=model, start=start, limit=limit, step=step)

    # Show graph
    # x = range(start, limit, step)
    # plt.plot(x, coherence_values)
    # plt.xlabel("Num Topics")
    # plt.ylabel("Coherence score")
    # plt.legend(("coherence_values"), loc='best')
    # plt.show()
    return best_model, best_num_of_topics

best_model, best_num_of_topics = find_best_num_of_topics(corpus, id2word, data_lemmatized, model)

def format_topics_sentences(ldamodel, corpus=corpus, texts=data):
    # Init output
    sent_topics_df = pd.DataFrame()
    topics = []
    keywords = {}
    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
                topics.append(int(topic_num))
                keywords[int(topic_num)] = topic_keywords
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return sent_topics_df, topics, keywords
print("Using best model with number of topics:", best_num_of_topics)
model = best_model
df_topic_sents_keywords , topics, keywords = format_topics_sentences(model, corpus=corpus, texts=data_lemmatized)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()


exports_folder = 'data/exports/'
timestamp = time.strftime("%Y%m%d-%H%M%S")
exports_filename = 'clustering_' + fileName + "_"+ timestamp + '.csv'
exports_filepath = os.path.join(exports_folder,exports_filename)
i=0

with open(exports_filepath,'w') as out:
    csv_out=csv.writer(out, delimiter = '|')
    csv_out.writerow(['label' ,'tweet_text' 'clean_text','keywords'])
    for tweet in tweets:
        tweet.set_cluster_label(topics[i])
        csv_out.writerow([tweet.cluster_label, tweet.tweet_text, tweet.clean_text, keywords[topics[i]]])
        i += 1
        pass

#df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

#df_dominant_topic.columns[""]
# Show
#df_dominant_topic.head(10)












