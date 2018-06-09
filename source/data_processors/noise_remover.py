import re
import string


from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

class NoiseRemover:

    stop = set(stopwords.words('english'))
    wordnet_lemmatizer = WordNetLemmatizer()

    def __init__(self):
        self.transformers = []
        self.transformers.append(self.clean_tweet)
        self.transformers.append(self.lemmatizer)
        pass

    def process_batch(self, tweets):
        result = []
        for tweet in tweets:
            result.append(self.process_tweet(tweet))
        return result

    def process_tweet(self, tweet):
        result = tweet
        for transformer in self.transformers:
            result = transformer(result)
        return result


    def lemmatizer(self, tweet):
        return tweet

    def clean_tweet(self, tweet):
        tweet = tweet.lower()
        tweet = re.sub(r'https?:\/\/.*[\r\n]*', ' ', tweet, flags=re.MULTILINE)
        tweet = re.sub(r'[Rr][tT][ ]?@[a-z0-9]*', ' ', tweet, flags=re.MULTILINE)
        tweet = re.sub(r'[#][a-z0-9A-Z]*', ' ', tweet, flags=re.MULTILINE)
        tweet = re.sub(r'@[a-z0-9]*', ' ', tweet, flags=re.MULTILINE)
        tweet = re.sub(r'[_"\-;%()|.,+&=*%]', '', tweet)
        tweet = re.sub(r'\.', ' . ', tweet)
        tweet = re.sub(r'\!', ' !', tweet)
        tweet = re.sub(r'\?', ' ?', tweet)
        tweet = re.sub(r'\,', ' ,', tweet)
        tweet = re.sub(r'd .c .', 'd.c.', tweet)
        tweet = re.sub(r'u .s .', 'd.c.', tweet)
        tweet = re.sub(r' amp ', ' and ', tweet)
        tweet = re.sub(r'pm', ' pm ', tweet)
        tweet = re.sub(r'news', ' news ', tweet)
        tweet = re.sub(r' . . . ', ' ', tweet)
        tweet = re.sub(r' .  .  . ', ' ', tweet)
        tweet = re.sub(r' ! ! ', ' ! ', tweet)
        tweet = re.sub(r'&amp', 'and', tweet)
        tweet = self.clean_sent(tweet)
        return tweet

    def clean_sent(self, sent):
        lem = self.wordnet_lemmatizer
        words = sent.replace(","," ").replace(";", " ").replace("#"," ").replace(":", " ").replace("@", " ").split()
        filtered_words = filter(lambda word: word.isalpha() and len(word) > 1 and word != "http" and word != "rt", [self.full_pipeline(lem, word) for word in words])
        return ' '.join(self.filter_stopwords(filtered_words))

    def filter_stopwords(self, words):
        return filter(lambda word: word not in self.stop, words)

    def full_pipeline(self, lem, word):
        word = word.lower()
        word = word.translate(string.punctuation)
        for val in ['a', 'v', 'n']:
            word = lem.lemmatize(word, pos=val)
        return word