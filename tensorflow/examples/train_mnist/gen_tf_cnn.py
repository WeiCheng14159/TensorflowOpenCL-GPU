# needed libraries
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

logs_path = '/tmp/tensorflow_logs/convnet'

# mnist.train = 55,000 input data
# mnist.test = 10,000 input data
# mnist.validate = 5,000 input data
mnist = input_data.read_data_sets("/tmp/tensorflow/mnist/input_data/", one_hot=True)

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

# our network!!!

g = tf.Graph()

with g.as_default():

	# input data
	x = tf.placeholder(tf.float32, shape=[None, 28*28], name='x')
	x_image = tf.reshape(x, [-1, 28, 28, 1])
	# correct labels
	y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y')

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
		y_f = tf.matmul(drop_layer, w_f) + b_f
		y_f_softmax = tf.nn.softmax(y_f)

	# loss
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_,
																  logits=y_f))

	# train step
	train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

	# accuracy
	correct_prediction = tf.equal(tf.argmax(y_f_softmax, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='test')

	# Create a summary to monitor loss tensor
	tf.summary.scalar("loss", loss)
	# Create a summary to monitor accuracy tensor
	tf.summary.scalar("accuracy", accuracy)
	# Merge all summaries into a single op
	merged_summary_op = tf.summary.merge_all()

	# init
	init = tf.variables_initializer(tf.global_variables(), name='init')

	# Running the graph

	num_steps = 100
	setp_to_test = 10
	batch_size = 32

	sess = tf.Session()

	sess.run(init)
	# op to write logs to Tensorboard
	summary_writer = tf.summary.FileWriter(logs_path,
										   graph=tf.get_default_graph())

	# Train the network for a small number of iterations
	# Then, finish the rest of the training process on mobile phone
	for step in range(num_steps):
		batch = mnist.train.next_batch(batch_size)

		ts, error, acc, summary = sess.run([train_step, loss, accuracy,
											merged_summary_op],
										   feed_dict={x: batch[0],
													  y_: batch[1],
													  keep_prob: 0.5})
		if step % setp_to_test == 0:
			train_accuracy = accuracy.eval({
				x: batch[0], y_: batch[1], keep_prob: 1.0}, sess)
			print('step %d, training accuracy %f' % (step, train_accuracy))

	tf.train.write_graph(sess.graph_def,
						'./',
						'mnist_100_cnn.pb', as_text=False)
