from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()


x = tf.placeholder(tf.float32, shape=[None, 784]) # m x 784 (28px img)
y_ = tf.placeholder(tf.float32, shape=[None, 10]) # m x Noutputs

W = tf.Variable(tf.zeros([784,10])) # weights
b = tf.Variable(tf.zeros([10])) # bias

sess.run(tf.global_variables_initializer())

# model
y = tf.matmul(x,W) + b #or use directly tf.nn.softmax( tf.matmul(x,W) + b )
# softmax => normalize(exp(x))


# TRAIN : logistic regression with stochastic gradient on cross entropy function 

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for i in range(1000):
	batch = mnist.train.next_batch(100)
	train_step.run(feed_dict={x: batch[0], y_: batch[1]})

#evaluate model
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
