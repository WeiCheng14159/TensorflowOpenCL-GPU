import tensorflow as tf

with tf.Session() as sess:
    x = tf.placeholder(tf.float32, [None, 1024], name="x")
    y = tf.placeholder(tf.float32, [1024 , None], name="y")

    result = tf.matmul( tf.matmul(x, y), y, name="matmul")

    tf.train.write_graph(sess.graph_def,
                         './',
                         'matmul.pb', as_text=False)
