import tensorflow as tf

n_classes = 2
num_of_units = 5
n_inputs = 300  # word vector dimension
n_steps = 50  # number of words fed to each RNN. We will feed the profile and tweet together

text_feature_length = n_steps * n_inputs

keep_prob = tf.placeholder(tf.float32)

best_accuracy_so_far = 0.0

y_ = tf.placeholder(tf.int32, shape=[None, n_classes])
rnn_X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
rnn_other_features = tf.placeholder(tf.float32, shape=[None,0])
keep_prob = tf.placeholder(tf.float32)
n_neurons = 50
learning_rate = 0.01