
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

import os
import csv

from parameters import Parameters

MAX_DOCUMENT_LENGTH = 300
EMBEDDING_SIZE = 5
HIDDEN_SIZE1 = 4
HIDDEN_SIZE2 = 4
ATTENTION_SIZE = 2
lr = 1e-2
BATCH_SIZE = 256
KEEP_PROB = 0.5
LAMBDA = 0.0001

MAX_LABEL = 2
epochs = 300

#dbpedia = tf.contrib.learn.datasets.load_dataset('dbpedia')
parameters = Parameters()
parameters.add_parameter("METHOD", "AVG-Word2Vec")
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

datafolder = 'data/classification_data/Training Data/1045'
exports_folder = 'data/exports/'
training_fileName = 'training_0.5.csv'
test_fileName = 'test.csv'
parameters.add_parameter("Data Folder", datafolder)
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

print("Train size: ", len(x_train))
print("Test size: ", len(x_test))
# split dataset to test and dev
x_test, x_dev, y_test, y_dev, dev_size, test_size, dev_tweets, test_tweets = \
    split_dataset(x_test, y_test, 0.1, test_tweets)
print("Validation size: ", dev_size)

#print(Y_TEST[0+dev_size:10+dev_size])
#print(y_test[0:10])

print(x_train[0,:])
print(x_test[0,:])
print(test_tweets[0])

graph = tf.Graph()
with graph.as_default():

    batch_x = tf.placeholder(tf.float32, [None, MAX_DOCUMENT_LENGTH])
    batch_y = tf.placeholder(tf.float32, [None, MAX_LABEL])
    keep_prob = tf.placeholder(tf.float32)

    #Layer - 1
    W1 = tf.Variable(tf.truncated_normal([MAX_DOCUMENT_LENGTH, HIDDEN_SIZE1], stddev=0.1))
    b1 = tf.Variable(tf.constant(0., shape=[HIDDEN_SIZE1]))
    h1 = tf.tanh(tf.matmul(batch_x,W1) + b1)
    drop1 = tf.nn.dropout(h1, keep_prob)

    #Layer - 2
    W2 = tf.Variable(tf.truncated_normal([HIDDEN_SIZE1, HIDDEN_SIZE2], stddev=0.1))
    b2 = tf.Variable(tf.constant(0., shape=[HIDDEN_SIZE2]))
    h2 = tf.tanh(tf.matmul(drop1, W2) + b2)
    drop2 = tf.nn.dropout(h2, keep_prob)


    W = tf.Variable(tf.truncated_normal([HIDDEN_SIZE2, MAX_LABEL], stddev=0.1))
    b = tf.Variable(tf.constant(0., shape=[MAX_LABEL]))

    y_hat = tf.nn.xw_plus_b(drop2, W, b)
    #print(y_hat.shape)



    # y_hat = tf.squeeze(y_hat)
    #y_hat = tf.subtract(y_hat[:,1],np.ones(y_hat[:,1].shape)*0.05)
    probability_penalty = 0.0
    modified_y_hat = tf.nn.softmax(y_hat)[:,1] - probability_penalty
    resultant_y_hat = tf.stack([tf.nn.softmax(y_hat)[:,0],modified_y_hat],axis=1)
    parameters.add_parameter("Optimizing Logit Variable", "resultant_y_hat")
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=resultant_y_hat, labels=batch_y))
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    # Accuracy metric
    prediction = tf.argmax(tf.nn.softmax(resultant_y_hat), 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(batch_y, 1)), tf.float32))

steps = 10001 # about 5 epochion.cpython-35.pyc
predictions = []
labels = []
with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    print("Initialized! ")

    print("Start trainning")
    start = time.time()
    for e in range(epochs):

        epoch_start = time.time()
        print("Epoch %d start !" % (e + 1))
        for x_batch, y_batch in fill_feed_dict(x_train, y_train, BATCH_SIZE):
            x_batch = x_batch.reshape(BATCH_SIZE, MAX_DOCUMENT_LENGTH)
            fd = {batch_x: x_batch, batch_y: y_batch, keep_prob: KEEP_PROB}
            l, _, acc = sess.run([loss, optimizer, accuracy], feed_dict=fd)
            #print(x_batch.shape)
            #print(y_batch)

        epoch_finish = time.time()
        x_batch = x_train.reshape(len(x_train), MAX_DOCUMENT_LENGTH)
        print("Training accuracy and loss: ", sess.run([accuracy, loss], feed_dict={
            batch_x: x_batch,
            batch_y: y_train,
            keep_prob: 1.0
        }))

        x_batch = x_dev.reshape(len(x_dev), MAX_DOCUMENT_LENGTH)
        val_acc_loss = "Validation accuracy and loss: ", sess.run([accuracy, loss], feed_dict={
            batch_x: x_batch,
            batch_y: y_dev,
            keep_prob: 1.0
        })
        print(val_acc_loss)
        validation_predictions = sess.run([prediction], feed_dict={
            batch_x: x_batch,
            batch_y: y_dev,
            keep_prob: 1.0
        })
        f1_predictions = np.array(validation_predictions)
        #print(f1_predictions.shape)
        f1_truelabels = np.argmax(y_dev, 1).reshape(-1,len(y_dev))
        #print(f1_truelabels.shape)
        f1score = f1_score(f1_truelabels, f1_predictions, average='macro')
        precision = precision_score(f1_truelabels, f1_predictions, average='macro')
        recall = recall_score(f1_truelabels, f1_predictions, average='macro')
        print("Validation Precision:{:.2} Recall:{:.2} F1:{:.2}".format(precision, recall, f1score))
        cnf_matrix = confusion_matrix(f1_truelabels[0,:], f1_predictions[0,:])

        print(cnf_matrix)


    parameters.add_parameter("Validation Statistics" ,"Precision:{:.2} Recall:{:.2} F1:{:.2} {}"
                             .format(precision, recall, f1score, val_acc_loss))
    parameters.add_parameter("Validation Confusion matrix", cnf_matrix)
    print("Training finished, time consumed : ", time.time() - start, " s")
    print("Start evaluating:  \n")
    cnt = 0
    test_acc = 0
    # for x_batch, y_batch in fill_feed_dict(x_test, y_test, BATCH_SIZE):
    #         fd = {batch_x: x_batch, batch_y: y_batch, keep_prob: 1.0}
    #         acc = sess.run(accuracy, feed_dict=fd)
    #         predictions.append(np.array(sess.run(prediction, feed_dict=fd)).reshape(-1,len(x_batch)))
    #         labels.append(y_batch)
    #         test_acc += acc
    #         cnt += 1
    fd = {batch_x: x_test.reshape(len(x_test), MAX_DOCUMENT_LENGTH), batch_y: y_test, keep_prob: 1.0}
    acc, predictions, probabilities = sess.run([accuracy, prediction, resultant_y_hat], feed_dict=fd)


    print("Test accuracy : %f %%" % ( acc))
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
    #print(predictions)

    parameters.add_parameter("probability_penalty", probability_penalty)
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
            csv_out.writerow([f1_predictions[i], f1_truelabels[i],probabilities[i][0],probabilities[i][1] , test_tweets[i]])



