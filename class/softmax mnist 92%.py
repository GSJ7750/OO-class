
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import numpy as np
import math


# In[2]:


tf.set_random_seed(0)


# In[3]:


mnist = read_data_sets("data", one_hot=True, reshape=False)


# In[4]:


X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y_ = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.random_uniform([784, 10]))
b = tf.Variable(tf.random_uniform([10]))


# In[5]:


XX=tf.reshape(X, [-1, 784])


# In[6]:


Y = tf.nn.softmax(tf.matmul(XX,W) + b)


# In[7]:


cross_entropy = -tf.reduce_mean(Y_*tf.log(Y))*1000


# In[8]:


correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))


# In[9]:


accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[10]:


optimizer = tf.train.GradientDescentOptimizer(0.005)
train = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


# In[11]:


for i in range(10001):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train, feed_dict={X: batch_xs, Y_: batch_ys})
    if i%1000 == 0:
        y, a, c = sess.run([Y, accuracy, cross_entropy], feed_dict={X: mnist.test.images, Y_: mnist.test.labels})
        print("accuacy:", a, "\n cost", c, "\n\n Y:", y, "\n")

