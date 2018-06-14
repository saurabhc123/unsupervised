import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np
import matplotlib.pyplot as plt
#from skimage.transform import resize
import scipy as sc
import pickle
import csv as csv
import datetime
import os as os
import random
import entities.placeholders as Placeholders
from torch.utils.data import DataLoader
from entities import tweet
from entities.tweets import Tweets
from data_processors.tweet_dataset import TweetDataSet
from data_processors.noise_remover import NoiseRemover
from entities.tweet import Tweet
from embeddings import word2vec
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score



class lstm_classifier():

    model_folder_name = "models/"
    model_filename = os.path.join(model_folder_name,"main_model.ckpt")

    def __init__(self, embeddings_generator):
        self.embeddings_generator = embeddings_generator
        pass

    def train(self, tweets_dataset, retrain):
        output_folder = 'data/exports/'
        if not os.path.exists(output_folder):
            print("Creating folder: " + output_folder)
            os.makedirs(output_folder)
        else:
            print("Folder exists: " + output_folder)

        word_vec = self.embeddings_generator
        with tf.variable_scope("LSTM"):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(Placeholders.n_neurons, forget_bias = 1.0)
            outputs, states = tf.nn.dynamic_rnn(lstm_cell, Placeholders.rnn_X, dtype=tf.float32)

        all_features = tf.concat([states[-1], Placeholders.rnn_other_features], 1)

        #fully_connected1_dropout = tf.nn.dropout(all_features, keep_prob=keep_prob)
        output_layer = tf.layers.dense(all_features, Placeholders.n_classes)
        fully_connected1_dropout = tf.nn.dropout(output_layer, keep_prob=Placeholders.keep_prob)
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= fully_connected1_dropout,
                                                                        labels=Placeholders.y_))

        loss = tf.reduce_mean(cross_entropy)
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

        correct_prediction = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Placeholders.y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        STEPS = 100
        MINIBATCH_SIZE = 10
        dataloader = DataLoader(tweets_dataset, batch_size= MINIBATCH_SIZE)
        with tf.Session() as sess:
            if os.path.exists(self.model_folder_name) & (not retrain):
                print("Model found in file: %s" % self.model_filename)
                saver.restore(sess, self.model_filename)
            else:
                if (retrain) & os.path.exists(self.model_folder_name):
                    print ("Retraining the model.")
                print ("Starting at:" , datetime.datetime.now())
                sess.run(tf.global_variables_initializer())
                print ("Initialization done at:" , datetime.datetime.now())
                for epoch in range(STEPS):
                    print ("Starting epoch", epoch, " at:", datetime.datetime.now())
                    for batch in dataloader:
                        tweets = Tweets(batch)
                        #tweets = list(tweets)
                        features = list(map(lambda tweet: tweet.get_embedding(),tweets))
                        labels = list(map(lambda tweet: tweet.get_class_label(),tweets))

                        rnn_features = np.array(features)\
                                        .reshape((-1, Placeholders.n_steps, Placeholders.n_inputs))
                        sess.run(train_step, feed_dict={Placeholders.rnn_X: rnn_features,
                                                        Placeholders.y_: labels,
                                                        Placeholders.keep_prob: 0.75})
                    if(epoch%10 == 0):
                        mse = loss.eval(feed_dict={Placeholders.rnn_X: rnn_features,
                                                    Placeholders.y_: labels,
                                                    Placeholders.keep_prob: 1.0})
                        print("Iter " + str(epoch) + ", Minibatch Loss= " + \
                              "{:.6f}".format(mse))
                        # train_accuracy = test_all(sess, accuracy, kdm.train, fc7, word_vec, output_layer, correct_prediction, loss, kdm, epoch, datasetType="Train")
                        # validation_accuracy = test_all(sess, accuracy, kdm.validation, fc7, word_vec, output_layer, correct_prediction, loss, kdm, epoch, datasetType="Validation")
                        # if (validation_accuracy > Placeholders.best_accuracy_so_far):
                        #     Placeholders.best_accuracy_so_far = validation_accuracy
                        #     test_all(sess, accuracy, kdm.test, fc7, word_vec, output_layer, correct_prediction, loss, kdm, epoch)
                        # elif (train_accuracy > 70):
                        #     test_all(sess, accuracy, kdm.test, fc7, word_vec, output_layer, correct_prediction, loss, kdm, epoch)
                # test_all(sess, accuracy, kdm.test, fc7, word_vec, output_layer, correct_prediction, loss, kdm)
                save_path = saver.save(sess, self.model_filename)
                print("Model saved in file: %s" % save_path)
        return accuracy


    def test(self, tweets):
        pass


datafolder = 'data/classification_data/'
exports_folder = 'data/exports/'
fileName = 'Dataset_z_1024_tweets.json'
embedding_generator = word2vec.word2vec()
noise_remover = NoiseRemover()
#fileName = 'junk.json'
filepath = os.path.join(datafolder,fileName)
dataset = TweetDataSet(datafolder, fileName, vectorizer=embedding_generator, transformer=noise_remover)
classifier = lstm_classifier(embedding_generator)
classifier.train(dataset, True)