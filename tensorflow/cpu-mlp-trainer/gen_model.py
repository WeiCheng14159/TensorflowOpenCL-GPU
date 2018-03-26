import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32, [None, 256], name="x")
y = tf.placeholder(tf.float32, [None, 10], name="y")

w1 = tf.Variable(tf.truncated_normal([256, 128], stddev=0.1))
b1 = tf.Variable(tf.constant(0.0, shape=[128]))

w2 = tf.Variable(tf.truncated_normal([128, 10], stddev=0.1))
b2 = tf.Variable(tf.constant(0.0, shape=[10]))

a = tf.nn.tanh(tf.nn.bias_add(tf.matmul(x, w1), b1))
y_out = tf.nn.tanh(tf.nn.bias_add(tf.matmul(a, w2), b2), name="y_out")
cost = tf.reduce_sum(tf.square(y-y_out), name="cost")
optimizer = tf.train.AdamOptimizer(use_locking=True).minimize(cost, name="train")

init = tf.initialize_variables(tf.all_variables(), name='init_all_vars_op')

sess = tf.Session()
sess.run(init)

tf.train.write_graph(sess.graph_def,
                     './',
                     'mlp.pb', as_text=False)

target_x = np.random.rand(100,256)
target_y = np.random.rand(100,10)

initial_cost = sess.run(cost, feed_dict={ x: target_x, y: target_y })
print( "initial cost %f" % initial_cost )
for step in range(1000):
    if step % 10 == 0:
        c = sess.run(cost, feed_dict={ x: target_x, y: target_y })
        print('step %d, cost %f' % (step, c))
    optimizer.run(feed_dict={ x: target_x, y: target_y }, session=sess)
