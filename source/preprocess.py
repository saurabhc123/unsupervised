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
import numpy as np
import csv
import time

datafolder = 'data/classification_data/'
exports_folder = 'data/exports/'
fileName = 'Dataset_z_1024_tweets.json'
exports_filename = 'clustering_' + time.strftime("%Y%m%d-%H%M%S") + '.csv'
#fileName = 'junk.json'
filepath = os.path.join(datafolder,fileName)
exports_filepath = os.path.join(exports_folder,exports_filename)

dataset = TweetDataSet(datafolder, fileName)
dataloader = DataLoader(dataset, batch_size=1)
tweets = []
sentence_vectors = []
wv = word2vec.word2vec()
for data in dataloader:
    tweet = Tweet(data)
    sentence_vector, clean_text = wv.get_sentence_vector(tweet.tweet_text)
    tweet.set_clean_text(clean_text)
    tweets.append(tweet)
    sentence_vectors.append(sentence_vector)
    #pprint(tweet.tweet_text)

print(len(tweets))
sentence_vectors = np.array(sentence_vectors).reshape(-1,300)
print (sentence_vectors.shape)

clusterer = DBScan.DBScanClusterer()
labels = clusterer.perform_clustering(sentence_vectors)
i = 0

with open(exports_filepath,'w') as out:
    csv_out=csv.writer(out, delimiter = '|')
    csv_out.writerow(['label' , 'tweet_text'])
    for tweet in tweets:
        tweet.label = labels[i]
        csv_out.writerow([tweet.label, tweet.clean_text, tweet.tweet_text])
        i += 1

for i in range(50):
    print(tweets[i].label ,tweets[i].tweet_text)

print(len(tweets))

