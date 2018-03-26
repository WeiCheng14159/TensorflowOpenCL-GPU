# needed libraries
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Implementing Convnet with TF
def weight_variable(shape, name=None):
    if name:
        w = tf.truncated_normal(shape, stddev=0.1, name=name)
    else:
        w = tf.truncated_normal(shape, stddev=0.1)

    return tf.Variable(w)

def bias_variable(shape, name=None):
    # avoid dead neurons
    if name:
        b = tf.constant(0.1, shape=shape, name=name)
    else:
        b = tf.constant(0.1, shape=shape)
    return tf.Variable(b)

# Pooling
def max_pool_2x2(X):
    return tf.nn.max_pool(X, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def new_conv_layer(X, w):
    return tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='SAME')

def main(_):

    # Import data
    mnist = input_data.read_data_sets(FLAGS.mnistDataDir, one_hot=True)

    # Training parameters
    maxEpochs = FLAGS.maxEpochs
    batchSize = FLAGS.batchSize
    testStep = FLAGS.testStep

    # Network parameters
    n_input = 784 # MNIST data input (img shape: 28*28)
    n_classes = 10 # MNIST total classes (0-9 digits)

    # Create the model
    X = tf.placeholder(tf.float32, shape=[None, n_input ], name='input')
    Y = tf.placeholder(tf.float32, shape=[None, n_classes], name='output')
    xImage = tf.reshape(X, [-1, 28, 28, 1])

    # fist conv layer
    with tf.name_scope('convLayer1'):
        w1 = weight_variable([5, 5, 1, 32])
        b1 = bias_variable([32])
        convlayer1 = tf.nn.relu(new_conv_layer(xImage, w1) + b1)
        max_pool1 = max_pool_2x2(convlayer1)

    # second conv layer
    with tf.name_scope('convLayer2'):
        w2 = weight_variable([5, 5, 32, 64])
        b2 = bias_variable([64])
        convlayer2 = tf.nn.relu(new_conv_layer(max_pool1, w2) + b2)
        max_pool2 = max_pool_2x2(convlayer2)

    # flat layer
    with tf.name_scope('flattenLayer'):
        flat_layer = tf.reshape(max_pool2, [-1, 7 * 7 * 64])

    # fully connected layer
    with tf.name_scope('FullyConnectedLayer'):
        wfc1 = weight_variable([7 * 7 * 64, 1024])
        bfc1 = bias_variable([1024])
        fc1 = tf.nn.relu(tf.matmul(flat_layer, wfc1) + bfc1)

    # DROPOUT
    with tf.name_scope('Dropout'):
        keep_prob = tf.placeholder(tf.float32)
        drop_layer = tf.nn.dropout(fc1, keep_prob)

    # final layer
    with tf.name_scope('FinalLayer'):
        w_f = weight_variable([1024, 10])
        b_f = bias_variable([10])
        y_out = tf.matmul(drop_layer, w_f) + b_f
        yFinalSoftmax = tf.nn.softmax(y_out)

    # Define loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=yFinalSoftmax, labels=Y))

    # Define optimizer
    with tf.name_scope('adam_optimizer'):
        train_op = tf.train.AdamOptimizer().minimize(loss, name="train")

    # Define accuracy
    prediction = tf.equal(tf.argmax(yFinalSoftmax, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32), name='test')

    # Create a summary to monitor cross_entropy tensor
    tf.summary.scalar("loss", loss)
    # Create a summary to monitor accuracy tensor
    tf.summary.scalar("accuracy", accuracy)
    # Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()

    # Initializing the variables
    init = tf.initialize_variables(tf.all_variables(), name='init')

    with tf.Session() as sess:
        # Session Init
        sess.run(init)

        # Logger Init
        summaryWriter = tf.summary.FileWriter(FLAGS.logDir, graph=sess.graph)

        # Training
        for step in range( maxEpochs ):
            # Get MNIST training data
            batchImage, batchLabel = mnist.train.next_batch(batchSize)

            # Test training model for every testStep
            if step % testStep == 0:
                # Run accuracy op & summary op to get accuracy & training progress
                acc, summary = sess.run( [ accuracy, merged_summary_op ], \
                    feed_dict={ X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0})

                # Write accuracy to log file
                summaryWriter.add_summary(summary, step)

                # Print accuracy
                print('step %d, training accuracy %f' % (step, acc))

            # Run training op
            train_op.run(feed_dict={ X: batchImage, Y: batchLabel, keep_prob: 0.5})

        # Write TF model
        tf.train.write_graph(sess.graph_def,
                            './',
                            'mnist_cnn.pb', as_text=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mnistDataDir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='MNIST data directory')
    parser.add_argument('--logDir', type=str, default='/tmp/tensorflow_logs/convnet',
                        help='Training progress data directory')
    parser.add_argument('--batchSize', type=int, default=50,
                        help='Training batch size')
    parser.add_argument('--maxEpochs', type=int, default=10000,
                        help='Maximum training steps')
    parser.add_argument('--testStep', type=int, default=100,
                        help='Test model accuracy for every testStep iterations')
    FLAGS, unparsed = parser.parse_known_args()
    # Program entry
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
