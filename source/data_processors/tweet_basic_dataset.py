from torch.utils.data import Dataset
import json
import os
from entities.tweet import Tweet
import numpy as np
import csv

class TweetBasicDataSet(Dataset):

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
        try:
            cluster_label = int(row[0])
            tweet_text = clean_text = row[1]
            if len(row) > 2:
                tweet_text = row[2]
            else:
                tweet_text =  clean_text
        except:
            print("Error ->", idx)
        if self.transformer is not None:
            clean_text = self.transformer.process_tweet(tweet_text)
        if self.vectorizer is not None:
            clean_text, _ = self.vectorizer.get_sentence_vector(clean_text)
            #print(clean_text)

        return {'text': tweet_text, 'clean_text': clean_text, 'cluster_label': cluster_label}
        #return {'obj':self.tweet}
        #return Tweet(tweet_text,tweet_instance["id"]), vectorized_representation
