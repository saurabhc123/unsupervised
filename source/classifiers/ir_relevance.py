
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
import time

from torch.utils.data import DataLoader

from classifiers.prepare_data import *

# Hyperparameter
from data_processors.noise_remover import NoiseRemover
from data_processors.tweet_basic_dataset import TweetBasicDataSet
from data_processors.tweet_dataset import TweetDataSet
from data_processors.tweet_lda_dataset import TweetLDADataSet
from entities.tweet import Tweet
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

from parameters import Parameters
import os
import csv

MAX_DOCUMENT_LENGTH = 10
EMBEDDING_SIZE = 2
HIDDEN_SIZE = 4
ATTENTION_SIZE = 3
lr = 1e-3
BATCH_SIZE = 256
KEEP_PROB = 0.5
LAMBDA = 0.0001

MAX_LABEL = 2
epochs = 100

#dbpedia = tf.contrib.learn.datasets.load_dataset('dbpedia')
parameters = Parameters()
parameters.add_parameter("MAX_LABEL",MAX_LABEL)


# load data
x_train, y_train = ([],[])#load_data("data/classification_data/Training Data/train.csv", names=["Label", "clean_text", "tweet_text"])
x_test, y_test = ([],[])#load_data("data/classification_data/Training Data/test.csv")

datafolder = 'data/classification_data/Training Data/1045'
exports_folder = 'data/exports/'
training_fileName = 'training_large_top50_clusters.csv'
test_fileName = 'test.csv'
parameters.add_parameter("Data Folder", datafolder)
parameters.add_parameter("Training filename", training_fileName)
parameters.add_parameter("Test filename", test_fileName)
pre_processor = NoiseRemover()
training_dataset = TweetBasicDataSet(datafolder, training_fileName, transformer=pre_processor)
training_dataloader = DataLoader(training_dataset, batch_size=len(training_dataset.data))
test_dataset = TweetBasicDataSet(datafolder, test_fileName, transformer=pre_processor)
test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset.data))


training_tweets = []
tweets_dict = set()
sentence_vectors = []
test_tweets = []
X_TEST = []
TEST_TWEETS = []
Y_TEST = []

def to_one_hot(y, n_class):
    return np.eye(n_class)[y]

for data in training_dataloader:
    x_train = data['clean_text']
    y_train = to_one_hot(data['cluster_label'],MAX_LABEL)
    break

for data in test_dataloader:
    X_TEST = x_test = data['clean_text']
    TEST_TWEETS = test_tweets = data['text']
    Y_TEST = data['cluster_label']
    y_test = to_one_hot(data['cluster_label'],MAX_LABEL)
    break





# data preprocessing
# x_train, x_test, vocab, vocab_size = \
#     data_preprocessing(x_train, x_test, MAX_DOCUMENT_LENGTH)
# print(vocab_size)

#Create set out of (only positive) training data
positive_set = set()
for training_instance in x_train:
    #training_instance = training_instance.tolist()
    if training_instance not in positive_set:
        positive_set.add(training_instance)

#If the tweet words are in the training set, mark the class as positive, else negative
predictions = []
for test_instance in x_test:
    if test_instance in positive_set:
        predictions.append(1)
    else:
        predictions.append(0)


f1_predictions = np.array(predictions)
#print(f1_predictions.shape)
f1_truelabels = np.argmax(y_test, 1)
#print(f1_truelabels.shape)
f1score = f1_score(f1_truelabels, f1_predictions, average='macro')
precision = precision_score(f1_truelabels, f1_predictions, average='macro')
recall = recall_score(f1_truelabels, f1_predictions, average='macro')
print("Test Precision:{:.2} Recall:{:.2} F1:{:.2}".format(precision, recall, f1score))
cnf_matrix = confusion_matrix(f1_truelabels, f1_predictions)
print(cnf_matrix)


parameters.add_parameter("Test Statistics", "Precision:{:.2} Recall:{:.2} F1:{:.2}".format(precision, recall, f1score))
parameters.add_parameter("Test Confusion matrix", cnf_matrix)
exports_folder = 'data/exports/'
timestamp = time.strftime("%Y%m%d-%H%M%S")
parameters.write_parameters(exports_folder, timestamp+"_TestF1_{:.2}".format(f1score))


print("Identifier:{}".format(timestamp))
results_filename = "classification_results_" + timestamp + "_TestF1_{:.2}".format(f1score)+".csv"
filepath = os.path.join(exports_folder, results_filename)
with open(filepath,'w') as out:
    csv_out=csv.writer(out, delimiter = ',')
    csv_out.writerow(['Predicted' , 'Truth' ,'Text'])
    for i in range(len(predictions)):
        #print([f1_predictions[i], f1_truelabels[i], test_tweets[i]])
       csv_out.writerow([f1_predictions[i], f1_truelabels[i] , test_tweets[i]])



