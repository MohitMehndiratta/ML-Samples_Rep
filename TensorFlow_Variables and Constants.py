import tensorflow as tf

hello = tf.constant("hello")
world = tf.constant("world")

var1 = tf.constant(10)
var2 = tf.constant(30)

mat_1 = tf.fill((5, 5), 10, "my_matrix")

zeroes = tf.zeros((5, 5))
ones = tf.ones((5, 5))
randn=tf.random_normal((4,4),mean=0,stddev=1.0)
rando=tf.random_uniform((4,4),minval=0,maxval=1)

myops=[mat_1,zeroes,randn,rando,ones]

sess_new=tf.InteractiveSession()

with tf.Session() as sess:
    res = sess.run(hello + world)
    res1 = sess.run(var1 + var2)

for ops in myops:
    # print(sess_new.run(ops))
    # print(ops.eval())
    print("\n")


a=tf.constant([[1,2],[3,4]])
a.get_shape()

b=tf.constant([[10],[100]])
b.get_shape()

sess_n=tf.InteractiveSession()
# print(sess_n.run(a))
# print("\n")
# print(sess_n.run(b))


mul=tf.matmul(a,b)
print(sess_n.run(mul))

print(a,"\n",b)
# print(res, res1)
