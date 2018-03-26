# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def main(_):

    # Import data
    mnist = input_data.read_data_sets(FLAGS.mnistDataDir, one_hot=True)

    # Training parameters
    maxEpochs = FLAGS.maxEpochs
    batchSize = FLAGS.batchSize
    testStep = FLAGS.testStep

    # Network parameters
    n_hidden_1 = 50 # 1st layer number of neurons
    n_hidden_2 = 50 # 2nd layer number of neurons
    n_input = 784 # MNIST data input (img shape: 28*28)
    n_classes = 10 # MNIST total classes (0-9 digits)

    # tf Graph input
    X = tf.placeholder(tf.float32, [None, n_input], name="input")
    Y = tf.placeholder(tf.float32, [None, n_classes], name="output")

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Create model
    def multilayer_perceptron(x):
        # Hidden fully connected layer with `n_hidden_1` neurons
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        # Hidden fully connected layer with `n_hidden_2` neurons
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        # Output fully connected layer with a neuron for each class
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
        return out_layer

    # Construct model
    logits = multilayer_perceptron(X)

    # Define loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=Y))

    # Define optimizer
    with tf.name_scope('adam_optimizer'):
        train_op = tf.train.AdamOptimizer().minimize(loss, name="train")

    # Define accuracy
    prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32), name="test")

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
        summary_writer = tf.summary.FileWriter( FLAGS.logDir, graph=sess.graph)

        # Training
        for step in range(maxEpochs):
            # Get MNIST training data
            batchImage, batchLabel = mnist.train.next_batch(batchSize)

            # Test training model for every testStep
            if step % testStep == 0:
                # Run accuracy op & summary op to get accuracy & training progress
                acc, summary = sess.run([accuracy, merged_summary_op],
                    feed_dict={X: mnist.test.images, Y: mnist.test.labels})

                # Write accuracy to log file
                summary_writer.add_summary(summary, step)

                # Print accuracy
                print('step %d, training accuracy %f' % (step, acc))

            # Run training op
            train_op.run( feed_dict={ X: batchImage, Y: batchLabel }, session=sess)

        # Write TF model
        tf.train.write_graph(sess.graph_def,
                            './',
                            'mnist_mlp.pb', as_text=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mnistDataDir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='MNIST data directory')
    parser.add_argument('--logDir', type=str, default='/tmp/tensorflow_logs/mlpnet',
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
