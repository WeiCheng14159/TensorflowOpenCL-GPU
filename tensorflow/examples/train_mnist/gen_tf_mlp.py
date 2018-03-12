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

img_size = 28

def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir)

  # Create the model
  x = tf.placeholder(tf.float32, [None, img_size * img_size ], name="input")
  y = tf.placeholder(tf.int64,   [None                      ], name="output")

  W = tf.Variable(tf.truncated_normal([ img_size * img_size, 10], stddev=0.1), name="W")
  b = tf.Variable(tf.zeros([10]), name="b")

  y_out = tf.nn.tanh( tf.matmul(x, W) + b )

  # Define loss and optimizer
  cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_out)
  with tf.name_scope('adam_optimizer'):
      train_step = tf.train.AdamOptimizer().minimize(cross_entropy, name="train")

  correct_prediction = tf.equal(tf.argmax(y_out, 1), y)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="test")

  # Create a summary to monitor cross_entropy tensor
  tf.summary.scalar("cross_entropy", cross_entropy)
  # Create a summary to monitor accuracy tensor
  tf.summary.scalar("accuracy", accuracy)
  # Create a summary histogram for weight matrix
  tf.summary.histogram("W", W)
  # Create a summary histogram for biased vector
  tf.summary.histogram("b", b)

  # Merge all summaries into a single op
  merged_summary_op = tf.summary.merge_all()

  init_op = tf.initialize_variables(tf.all_variables(), name='init')

  sess = tf.Session()
  sess.run(init_op)

  summary_writer = tf.summary.FileWriter( FLAGS.log_dir, graph=sess.graph)

  # Train
  for step in range( FLAGS.max_steps ):
    batch_xs, batch_ys = mnist.train.next_batch( FLAGS.batch_size )
    if step % 100 == 0:
        acc, summary = sess.run([ accuracy, merged_summary_op ], \
            feed_dict={ x: mnist.test.images, y: mnist.test.labels })
        print('step %d, training accuracy %f' % (step, acc))
        summary_writer.add_summary(summary, step)
    train_step.run( feed_dict={ x: batch_xs, y: batch_ys }, session=sess)
  print('test accuracy %f' %  accuracy.eval(feed_dict={
      x: mnist.test.images, y: mnist.test.labels }, session=sess))
  tf.train.write_graph(sess.graph_def,
                        './',
                        'mnist_mlp.pb', as_text=False)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  parser.add_argument('--log_dir', type=str, default='/tmp/tensorflow_logs/mlpnet',
                      help='Directory for training data')
  parser.add_argument('--batch_size', type=int, default=50,
                      help='Batch size')
  parser.add_argument('--max_steps', type=int, default=2000,
                      help='Maximum training steps')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
