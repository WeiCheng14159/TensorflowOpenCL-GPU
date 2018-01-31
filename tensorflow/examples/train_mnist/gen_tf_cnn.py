# needed libraries
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

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

# pool
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def new_conv_layer(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

FLAGS = None
img_size = 28

def main(_):

    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    logs_path = FLAGS.log_dir

    # Create the model
    x = tf.placeholder(tf.float32, shape=[None, img_size * img_size ], name='input')
    x_image = tf.reshape(x, [-1, img_size, img_size, 1])
    y = tf.placeholder(tf.float32, shape=[None, 10], name='output')

    # fist conv layer
    with tf.name_scope('convLayer1'):
        w1 = weight_variable([5, 5, 1, 32])
        b1 = bias_variable([32])
        convlayer1 = tf.nn.relu(new_conv_layer(x_image, w1) + b1)
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
        y_f_softmax = tf.nn.softmax(y_out)

    # loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_out))

    # train step
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss, name="train")

    # accuracy
    correct_prediction = tf.equal(tf.argmax(y_f_softmax, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='test')

    # Create a summary to monitor loss tensor
    tf.summary.scalar("loss", loss)
    # Create a summary to monitor accuracy tensor
    tf.summary.scalar("accuracy", accuracy)
    # Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()

    # init
    init_op = tf.initialize_variables(tf.all_variables(), name='init')

    # Create session
    sess = tf.Session()

    # Init all variables
    sess.run(init_op)

    # Log writer for Tensorboard
    summary_writer = tf.summary.FileWriter(logs_path, graph=sess.graph)

    num_steps = 1000
    setp_to_test = 10
    batch_size = 100

    for step in range(num_steps):

        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        if step % setp_to_test == 0:

            summary = sess.run(merged_summary_op, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
            summary_writer.add_summary(summary, step)

            train_accuracy = accuracy.eval( { x: batch_xs, y: batch_ys, keep_prob: 1.0}, sess )
            print('step %d, training accuracy %g' % (step, train_accuracy))
        else:
            ts, summary = sess.run([ train_step, merged_summary_op ], \
                feed_dict={ x: batch_xs, y: batch_ys, keep_prob: 0.5})

    # Test trained model
    print('test accuracy %f' % sess.run(accuracy, feed_dict={x: mnist.test.images,
        y: mnist.test.labels, keep_prob: 1.0}) )

    tf.train.write_graph(sess.graph_def,
                        './',
                        'mnist_100_cnn.pb', as_text=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
    parser.add_argument('--log_dir', type=str, default='/tmp/tensorflow_logs/convnet',
                      help='Directory for training data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
