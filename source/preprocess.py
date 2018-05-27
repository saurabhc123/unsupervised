import json
import os
from pprint import pprint
from os import path
from data_processors.tweet_dataset import TweetDataSet
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from entities.tweet import Tweet

datafolder = 'data/classification_data/'
fileName = 'Dataset_z_1024_tweets.json'
#fileName = 'junk.json'
filepath = os.path.join(datafolder,fileName)

dataset = TweetDataSet(datafolder, fileName)
dataloader = DataLoader(dataset, batch_size=32)
for data in dataloader:
    tweet = Tweet(data)
    pprint(tweet.tweet_text)
