
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

MAX_DOCUMENT_LENGTH = 15
EMBEDDING_SIZE = 20
HIDDEN_SIZE = 16
ATTENTION_SIZE = 2
lr = 1e-3
BATCH_SIZE = 256
KEEP_PROB = 0.5
LAMBDA = 0.0001

MAX_LABEL = 2
epochs = 100

#dbpedia = tf.contrib.learn.datasets.load_dataset('dbpedia')

# load data
x_train, y_train = ([],[])#load_data("data/classification_data/Training Data/train.csv", names=["Label", "clean_text", "tweet_text"])
x_test, y_test = ([],[])#load_data("data/classification_data/Training Data/test.csv")

datafolder = 'data/classification_data/Training Data'
exports_folder = 'data/exports/'
training_fileName = 'train.csv'
test_fileName = 'test.csv'
pre_processor = NoiseRemover()
training_dataset = TweetBasicDataSet(datafolder, training_fileName, transformer=pre_processor)
training_dataloader = DataLoader(training_dataset, batch_size=len(training_dataset.data))
test_dataset = TweetBasicDataSet(datafolder, test_fileName, transformer=pre_processor)
test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset.data))


training_tweets = []
tweets_dict = set()
sentence_vectors = []

def to_one_hot(y, n_class):
    return np.eye(n_class)[y]

for data in training_dataloader:
    x_train = data['clean_text']
    y_train = to_one_hot(data['cluster_label'],MAX_LABEL)
    break

for data in test_dataloader:
    x_test = data['clean_text']
    y_test = to_one_hot(data['cluster_label'],MAX_LABEL)
    break




# data preprocessing
x_train, x_test, vocab, vocab_size = \
    data_preprocessing(x_train, x_test, MAX_DOCUMENT_LENGTH)
print(vocab_size)

# split dataset to test and dev
x_test, x_dev, y_test, y_dev, dev_size, test_size = \
    split_dataset(x_test, y_test, 0.1)
print("Validation size: ", dev_size)

graph = tf.Graph()
with graph.as_default():

    batch_x = tf.placeholder(tf.int32, [None, MAX_DOCUMENT_LENGTH])
    batch_y = tf.placeholder(tf.float32, [None, MAX_LABEL])
    keep_prob = tf.placeholder(tf.float32)

    embeddings_var = tf.Variable(tf.random_uniform([vocab_size, EMBEDDING_SIZE], -1.0, 1.0), trainable=True)
    batch_embedded = tf.nn.embedding_lookup(embeddings_var, batch_x)
    W = tf.Variable(tf.random_normal([HIDDEN_SIZE], stddev=0.1))
    print(batch_embedded.shape)  # (?, 256, 100)

    rnn_outputs, _ = bi_rnn(BasicLSTMCell(HIDDEN_SIZE), BasicLSTMCell(HIDDEN_SIZE),
                            inputs=batch_embedded, dtype=tf.float32)
    # Attention
    fw_outputs = rnn_outputs[0]
    # print(fw_outputs.shape)
    bw_outputs = rnn_outputs[1]
    H = fw_outputs + bw_outputs  # (batch_size, seq_len, HIDDEN_SIZE)
    M = tf.tanh(H) # M = tanh(H)  (batch_size, seq_len, HIDDEN_SIZE)
    # print(M.shape)
    # alpha (bs * sl, 1)
    alpha = tf.nn.softmax(tf.matmul(tf.reshape(M, [-1, HIDDEN_SIZE]), tf.reshape(W, [-1, 1])))
    r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(alpha, [-1, MAX_DOCUMENT_LENGTH, 1])) # supposed to be (batch_size * HIDDEN_SIZE, 1)
    # print(r.shape)
    r = tf.squeeze(r)
    h_star = tf.tanh(r) # (batch , HIDDEN_SIZE
    # attention_output, alphas = attention(rnn_outputs, ATTENTION_SIZE, return_alphas=True)


    drop = tf.nn.dropout(h_star, keep_prob)
    W = tf.Variable(tf.truncated_normal([HIDDEN_SIZE, MAX_LABEL], stddev=0.1))
    b = tf.Variable(tf.constant(0., shape=[MAX_LABEL]))
    y_hat = tf.nn.xw_plus_b(drop, W, b)
    #print(y_hat.shape)

    # y_hat = tf.squeeze(y_hat)
    #y_hat = tf.subtract(y_hat[:,1],np.ones(y_hat[:,1].shape)*0.05)
    modified_y_hat = tf.nn.softmax(y_hat)[:,1] - 0.5
    resultant_y_hat = tf.stack([tf.nn.softmax(y_hat)[:,0],modified_y_hat],axis=1)
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
            fd = {batch_x: x_batch, batch_y: y_batch, keep_prob: KEEP_PROB}
            l, _, acc = sess.run([loss, optimizer, accuracy], feed_dict=fd)
            #print(x_batch.shape)
            #print(y_batch)

        epoch_finish = time.time()
        print("Training accuracy and loss: ", sess.run([accuracy, loss], feed_dict={
            batch_x: x_train,
            batch_y: y_train,
            keep_prob: 1.0
        }))

        print("Validation accuracy and loss: ", sess.run([accuracy, loss], feed_dict={
            batch_x: x_dev,
            batch_y: y_dev,
            keep_prob: 1.0
        }))
        validation_predictions = sess.run([prediction], feed_dict={
            batch_x: x_dev,
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
        print("Validation Precision:{} Recall:{} F1:{}".format(precision, recall, f1score))
        cnf_matrix = confusion_matrix(f1_truelabels[0,:], f1_predictions[0,:])
        print(cnf_matrix)


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
    fd = {batch_x: x_test, batch_y: y_test, keep_prob: 1.0}
    acc, predictions = sess.run([accuracy, prediction], feed_dict=fd)


    print("Test accuracy : %f %%" % ( acc))
    f1_predictions = np.array(predictions)
    print(f1_predictions.shape)
    f1_truelabels = np.argmax(y_test, 1)
    print(f1_truelabels.shape)
    f1score = f1_score(f1_truelabels, f1_predictions, average='macro')
    precision = precision_score(f1_truelabels, f1_predictions, average='macro')
    recall = recall_score(f1_truelabels, f1_predictions, average='macro')
    print("Test Precision:{} Recall:{} F1:{}".format(precision, recall, f1score))
    cnf_matrix = confusion_matrix(f1_truelabels, f1_predictions)
    print(cnf_matrix)
    #print(predictions)




