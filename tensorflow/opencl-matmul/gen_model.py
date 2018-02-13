import tensorflow as tf

with tf.Session() as sess:
    x = tf.placeholder(tf.float32, [None, 784], name="x")
    y = tf.placeholder(tf.float32, [784 , None], name="y")

    result = tf.matmul(x, y, name="matmul")

    tf.train.write_graph(sess.graph_def,
                         './',
                         'matmul.pb', as_text=False)
