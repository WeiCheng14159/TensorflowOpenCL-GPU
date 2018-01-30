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

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784], name="input")
  y = tf.placeholder(tf.int64, [None], name="output")

  W = tf.Variable(tf.truncated_normal([784, 10], stddev=0.1))
  b = tf.Variable(tf.zeros([10]))

  y_out = tf.matmul(x, W) + b

  # Define loss and optimizer
  cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_out)
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy, name="train")

  init_op = tf.initialize_variables(tf.all_variables(), name='init_all_vars_op')

  sess = tf.Session()
  sess.run(init_op)

  batch_size=1000
  # Train
  for _ in range(100):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y_out, 1), y)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y: mnist.test.labels}))

  tf.train.write_graph(sess.graph_def,
  						'./',
  						'mnist_100_mlp.pb', as_text=False)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
