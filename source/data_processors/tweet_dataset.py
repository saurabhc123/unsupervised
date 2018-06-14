from torch.utils.data import Dataset
import json
import os
from entities.tweet import Tweet
import numpy as np

class TweetDataSet(Dataset):
    def __init__(self, datafolder, filename, transformer=None, vectorizer=None):
        filepath = os.path.join(datafolder,filename)
        self.transformer = transformer
        self.vectorizer = vectorizer
        with open(filepath) as f:
            content = f.read()
            content = content.replace("][",",")
            self.data = json.loads("".join(content))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tweet_instance = self.data[idx]
        tweet_text = tweet_instance['text'].replace('\n', ' ').replace('\r', '')
        vectorized_representation = np.array(10)
        if self.transformer is not None:
            tweet_text = self.transformer.transform(tweet_text)
        if self.vectorizer is not None:
            vectorized_representation = self.vectorizer.get_sentence_vector(tweet_text)

        return {'text': tweet_text, 'embedding': vectorized_representation, 'id':tweet_instance["id"]}
        #return Tweet(tweet_text,tweet_instance["id"]), vectorized_representation
