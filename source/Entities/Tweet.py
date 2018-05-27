class Tweet():
    def __init__(self, tweet_text, tweet_id):
        self.tweet_text = tweet_text
        self.tweet_id = tweet_id

    def __init__(self, dict):
        self.tweet_text = dict['text']
        self.tweet_id = dict['id']