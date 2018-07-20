
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
from embeddings.word2vec import word2vec
from entities.tweet import Tweet
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.decomposition import PCA

from parameters import Parameters

MAX_DOCUMENT_LENGTH = 300
EMBEDDING_SIZE = 5
HIDDEN_SIZE1 = 4
HIDDEN_SIZE2 = 4
ATTENTION_SIZE = 2
lr = 1e-4
BATCH_SIZE = 256
KEEP_PROB = 0.5
LAMBDA = 0.0001

MAX_LABEL = 2
epochs = 200

#dbpedia = tf.contrib.learn.datasets.load_dataset('dbpedia')
parameters = Parameters()
parameters.add_parameter("METHOD", "O-SVM")
parameters.add_parameter("MAX_DOCUMENT_LENGTH", MAX_DOCUMENT_LENGTH)
parameters.add_parameter("EMBEDDING_SIZE",EMBEDDING_SIZE)
parameters.add_parameter("HIDDEN_SIZE1",HIDDEN_SIZE1)
parameters.add_parameter("HIDDEN_SIZE2",HIDDEN_SIZE2)
parameters.add_parameter("lr",lr)
parameters.add_parameter("BATCH_SIZE",BATCH_SIZE)
parameters.add_parameter("KEEP_PROB",KEEP_PROB)
parameters.add_parameter("LAMBDA",LAMBDA)
parameters.add_parameter("MAX_LABEL",MAX_LABEL)
parameters.add_parameter("epochs",epochs)

# load data
x_train, y_train = ([],[])#load_data("data/classification_data/Training Data/train.csv", names=["Label", "clean_text", "tweet_text"])
x_test, y_test = ([],[])#load_data("data/classification_data/Training Data/test.csv")

datafolder = 'data/classification_data/Training Data'
exports_folder = 'data/exports/'
training_fileName = 'training_one_class.csv'
test_fileName = 'test.csv'
parameters.add_parameter("Training filename", training_fileName)
parameters.add_parameter("Test filename", test_fileName)
pre_processor = NoiseRemover()
vectorizer = word2vec()
training_dataset = TweetBasicDataSet(datafolder, training_fileName, transformer=pre_processor, vectorizer = vectorizer)
training_dataloader = DataLoader(training_dataset, batch_size=len(training_dataset.data))
test_dataset = TweetBasicDataSet(datafolder, test_fileName, transformer=pre_processor, vectorizer=vectorizer)
test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset.data))


training_tweets = []
tweets_dict = set()
sentence_vectors = []


def to_one_hot(y, n_class):
    return np.eye(n_class)[y]

for data in training_dataloader:
    x_train = data['clean_text']
    y_train = to_one_hot(data['cluster_label'], MAX_LABEL)
    break

for data in test_dataloader:
    x_test = data['clean_text']
    y_test = to_one_hot(data['cluster_label'], MAX_LABEL)
    break

x_train = np.array(x_train).reshape(len(x_train), MAX_DOCUMENT_LENGTH)
x_test = np.array(x_test).reshape(len(x_test), MAX_DOCUMENT_LENGTH)

pca = PCA(n_components=10, whiten=True)
pca = pca.fit(x_train)
print('Explained variance percentage = %0.2f' % sum(pca.explained_variance_ratio_))
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)

print("Train size: ", x_train.shape)
print("Test size: ", x_test.shape)
# split dataset to test and dev
x_test, x_dev, y_test, y_dev, dev_size, test_size = \
    split_dataset(x_test, y_test, 0.1)
print("Validation size: ", dev_size)

gamma_values = [1.0,0.1,0.01,0.001, 0.0001, 0.0001]
nu_values = [0.05,0.08, 0.1, 0.2, 0.25, 0.3]

best_f1_validation = 0.0
best_model = []

for gamma in gamma_values:
    for nu in nu_values:
        oc_svm_clf = svm.OneClassSVM(gamma=gamma, kernel='rbf', nu=nu)
        oc_svm_clf.fit(x_train)
        predictions = oc_svm_clf.predict(x_dev)
        predictions = [0 if prediction < 1 else 1 for prediction in predictions]
        f1_predictions = np.array(predictions)
        f1_truelabels = np.argmax(y_dev, 1)
        f1score = f1_score(f1_truelabels, f1_predictions, average='macro')
        print("Validation F1:{} achieved for gamma:{} and nu:{}.".format(f1score, gamma, nu))
        if f1score > best_f1_validation:
            best_f1_validation = f1score
            best_model = oc_svm_clf
            print("Best validation F1:{} achieved for gamma:{} and nu:{}".format(f1score, gamma, nu))


oc_svm_clf = best_model

predictions = oc_svm_clf.predict(x_test)
predictions = [0 if prediction < 1 else 1 for prediction in predictions]

f1_predictions = np.array(predictions)
#print(f1_predictions.shape)
f1_truelabels = np.argmax(y_test, 1)
#print(f1_truelabels.shape)
f1score = f1_score(f1_truelabels, f1_predictions, average='macro')
precision = precision_score(f1_truelabels, f1_predictions, average='macro')
recall = recall_score(f1_truelabels, f1_predictions, average='macro')
print("Test Precision:{} Recall:{} F1:{}".format(precision, recall, f1score))
cnf_matrix = confusion_matrix(f1_truelabels, f1_predictions)
print(cnf_matrix)
#print(predictions)


parameters.add_parameter("Test Statistics", "Precision:{} Recall:{} F1:{}".format(precision, recall, f1score))
parameters.add_parameter("Test Confusion matrix", cnf_matrix)
exports_folder = 'data/exports/'
timestamp = time.strftime("%Y%m%d-%H%M%S")
parameters.write_parameters(exports_folder, timestamp+"_TestF1_{:.4}".format(f1score))



