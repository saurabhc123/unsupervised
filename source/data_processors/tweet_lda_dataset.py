from torch.utils.data import Dataset
import json
import os
from entities.tweet import Tweet
import numpy as np
import csv

class TweetLDADataSet(Dataset):

    def __init__(self, datafolder, filename, transformer=None, vectorizer=None):
        filepath = os.path.join(datafolder,filename)
        self.transformer = transformer
        self.vectorizer = vectorizer
        self.data = []
        with open(filepath) as csvfile:
            rowdata = csv.reader(csvfile, delimiter='|')
            next(rowdata, None)  # skip the headers
            self.data = list(rowdata)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tweet_instance = self.data[idx]
        row = tweet_instance
        cluster_label = row[0]
        tweet_text = row[1]
        clean_text = row[2]
        #probs = row[3]
        # self.tweet = Tweet()
        # self.tweet.set_cluster_label(cluster_label)
        # self.tweet.set_clean_text(clean_text)

        vectorized_representation = np.array(10)
        if self.vectorizer is not None:
            vectorized_representation = self.vectorizer.get_sentence_vector(tweet_instance.tweet_text)

        return {'text': tweet_text, 'embedding': vectorized_representation, 'id':0, 'clean_text': clean_text, 'cluster_label': cluster_label}
        #return {'obj':self.tweet}
        #return Tweet(tweet_text,tweet_instance["id"]), vectorized_representation
