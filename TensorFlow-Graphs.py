import tensorflow as tf

x = tf.constant([[100, 100]])

var1 = tf.constant(10)
var2 = tf.constant(20)

var3 = var1 * var2

with tf.Session() as sess:
    res = sess.run(var3)

g1 = tf.Graph()

with g1.as_default():
    print(g1 is tf.get_default_graph())  # -- will return TRUE as it is a default graph in this context

print(g1 is tf.get_default_graph) # -- will return FALSE as it is a default graph in this above context only

# print(tf.get_default_graph)
# print(tf.get_default_session)
