
import tensorflow as tf

from embeddings.word2vec import word2vec
import  datetime
from os import listdir
from os.path import isfile, join
import numpy as np

numDimensions = 300
maxSeqLength = 250
batchSize = 24
lstmUnits = 64
numClasses = 2
iterations = 100000


vectorizer = word2vec()

positiveFiles = ['data/sentiment/positiveReviews/' + f for f in listdir('data/sentiment/positiveReviews/') if isfile(join('data/sentiment/positiveReviews/', f))]
negativeFiles = ['data/sentiment/negativeReviews/' + f for f in listdir('data/sentiment/negativeReviews/') if isfile(join('data/sentiment/negativeReviews/', f))]
numWords = []
examples = []
for pf in positiveFiles:
    with open(pf, "r", encoding='utf-8') as f:
        line=f.readline()
        examples.append(line)
        counter = len(line.split())
        numWords.append(counter)
print('Positive files finished')

for nf in negativeFiles:
    with open(nf, "r", encoding='utf-8') as f:
        line=f.readline()
        examples.append(line)
        counter = len(line.split())
        numWords.append(counter)
print('Negative files finished')

numFiles = len(numWords)
print('The total number of files is', numFiles)
print('The total number of words in the files is', sum(numWords))
print('The average number of words in the files is', sum(numWords)/len(numWords))

#wordVectors = np.load('data/sentiment/idsMatrix.npy')

from random import randint

def getTrainBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength, numDimensions])
    for i in range(batchSize):
        if (i % 2 == 0):
            num = randint(1,11499)
            labels.append([1,0])
        else:
            num = randint(13499,24999)
            labels.append([0,1])
        arr[i] = vectorizer.get_sentence_matrix(examples[num-1:num][0])
    return arr, labels

def getTestBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength, numDimensions])
    for i in range(batchSize):
        num = randint(11499,13499)
        if (num <= 12499):
            labels.append([1,0])
        else:
            labels.append([0,1])
        arr[i] = vectorizer.get_sentence_matrix(examples[num-1:num][0])
    return arr, labels

def print_sentiment(predictedSentiment):
    if (predictedSentiment[0] > predictedSentiment[1]):
        print("Positive Sentiment")
    else:
        print("Negative Sentiment")
tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.float32, [batchSize, maxSeqLength, numDimensions])

#data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
#data = np.array(tf.nn.embedding_lookup(wordVectors,input_data), dtype=np.float32)

lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.25)
value, _ = tf.nn.dynamic_rnn(lstmCell, input_data, dtype=tf.float32)

weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)

correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))



sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)
for i in range(iterations):
   #Next Batch of reviews
   nextBatch, nextBatchLabels = getTrainBatch();
   sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})

   #Write summary to Tensorboard
   if (i % 50 == 0):
       summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
       writer.add_summary(summary, i)

   #Save the network every 10,000 training iterations
   if (i % 10000 == 0 and i != 0):
       acc, l = sess.run([accuracy, loss], {input_data: nextBatch, labels: nextBatchLabels})
       print("Training Statistics:Accuracy:{} Loss:{}".format(acc, l))
       save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
       print("saved to %s" % save_path)

writer.close()
# inputText = "That movie was terrible."
# inputMatrix = vectorizer.get_sentence_matrix(inputText)
# predictedSentiment = sess.run(prediction, {input_data: inputMatrix})[0]
# print_sentiment()
#
# inputText = "That movie was the best one I have ever seen."
# inputMatrix = vectorizer.get_sentence_matrix(inputText)
# predictedSentiment = sess.run(prediction, {input_data: inputMatrix})[0]
# print_sentiment()


