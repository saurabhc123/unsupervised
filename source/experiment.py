
from abc import ABC, abstractmethod
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
from parameters import Parameters


class Experiment():



    def __init__(self, name, dataloader, preprocesor, embedding_generator, clusterer, exports_filepath, parameters):
        self.name = name
        self.data = dataloader
        self.preprocessor = preprocesor
        self.embedding_generator = embedding_generator
        self.clusterer = clusterer
        self.exports_filepath = exports_filepath
        self.dataloader = dataloader
        self.parameters = parameters
        pass


    def perform_experiment(self):
        print ("Performing experiment:" , self.name)
        tweets = []
        tweets_dict = set()
        sentence_vectors = []
        wv = self.embedding_generator
        for data in self.dataloader:
            tweet = Tweet(data)
            clean_text = self.preprocessor.process_tweet(tweet.tweet_text)
            sentence_vector, _ = wv.get_sentence_vector(clean_text)
            if clean_text not in tweets_dict:
                tweets_dict.add(clean_text)
            else:
                continue
            tweet.set_clean_text(clean_text)
            tweets.append(tweet)
            sentence_vectors.append(sentence_vector)
            #pprint(tweet.tweet_text)

        print(len(tweets))
        sentence_vectors = np.array(sentence_vectors).reshape(-1,300)
        print (sentence_vectors.shape)

        labels = self.clusterer.perform_clustering(sentence_vectors, self.parameters)
        i = 0

        with open(self.exports_filepath,'w') as out:
            csv_out=csv.writer(out, delimiter = '|')
            csv_out.writerow(['label' , 'tweet_text'])
            for tweet in tweets:
                tweet.label = labels[i]
                csv_out.writerow([tweet.label, tweet.clean_text, tweet.tweet_text])
                i += 1
                pass

