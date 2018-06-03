class Tweet():
    def __init__(self, tweet_text, tweet_id):
        self.tweet_text = tweet_text
        self.tweet_id = tweet_id


    def __init__(self, dict):
        self.tweet_text = "".join(dict['text'])
        self.tweet_id = dict['id']

    def set_label(self,label):
        self.label = label

    def set_clean_text(self,clean_text):
        self.clean_text = clean_text
