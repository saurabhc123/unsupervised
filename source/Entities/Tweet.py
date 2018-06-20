class Tweet():
    def __init__(self, tweet_text, tweet_id):
        self.tweet_text = tweet_text
        self.tweet_id = tweet_id

    def __init__(self, tweet_text, tweet_id, class_label, embedding=None):
        self.__init__(tweet_text, tweet_id)
        self.class_label = class_label
        if embedding is not None:
            self.tweet_embedding = embedding

    def __init__(self, dict):
        # if dict.contains('obj'):
        #     objTweet = dict['obj']
        #     self.tweet_text = objTweet.tweet_text
        #     self.clean_text = objTweet.clean_text
        #     self.cluster_label = objTweet.cluster_label
        #     pass
        self.tweet_text = "".join(dict['text'])
        self.tweet_id = dict['id']
        if 'embedding' in dict.keys():
            self.tweet_embedding = dict['embedding']
            if type(self.tweet_embedding) is list:
                self.tweet_embedding = self.tweet_embedding[0]
        if 'clean_text' in dict.keys():
            self.clean_text = dict['clean_text']
            if type(self.clean_text) is list:
                self.clean_text = self.clean_text[0]
        if 'cluster_label' in dict.keys():
            self.cluster_label = dict['cluster_label']
            if type(self.cluster_label) is list:
                self.cluster_label = self.cluster_label[0]

    def set_cluster_label(self, cluster_label):
        self.cluster_label = cluster_label

    def set_class_label(self, class_label):
        self.class_label = class_label

    def get_class_label(self):
        return 1.0
        #return self.class_label

    def set_clean_text(self,clean_text):
        self.clean_text = clean_text

    def get_word_count(self):
        words = self.clean_text.split(' ')
        return len(words)

    def get_embedding(self):
        return self.tweet_embedding
