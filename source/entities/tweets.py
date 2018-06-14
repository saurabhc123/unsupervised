from entities.tweet import Tweet

class Tweets:
    def __init__(self, dict):
        #get the list
        embeddings = dict['embedding']
        texts = dict['text']
        tweets = []
        for i in range(len(embeddings)):
            temp_dict = {'text': texts[i], 'embedding': embeddings[i], 'id':'0'}
            tweet = Tweet(temp_dict)
            tweet.set_class_label(0)
            tweets.append(tweet)
        self.tweets = tweets

    def get_tweets(self):
        return self.tweets
