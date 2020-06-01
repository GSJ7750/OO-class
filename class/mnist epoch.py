
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import pandas as pd


# In[11]:


mnist = read_data_sets("data", one_hot=True, reshape=False)


# In[3]:


epoch = 15
batch_size =100


# In[12]:


X1 = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
X2 = tf.reshape(X1, [-1, 784])
Y = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.random_uniform([784, 10]))
b = tf.Variable(tf.random_uniform([10]))


# In[13]:


logits = tf.matmul(X2, W) + b
H = tf.nn.softmax(logits)


# In[14]:


cost = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y)


# In[15]:


is_correct = tf.equal(tf.argmax(Y,1), tf.argmax(H,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))


# In[16]:


optimizer = tf.train.AdamOptimizer(0.005)
train = optimizer.minimize(cost)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


# In[18]:


for e in range(epoch):
    total_batch = int(mnist.train.num_examples/batch_size)
    print("epoch = ", e+1)
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train, feed_dict={X1: batch_xs, Y: batch_ys})
        if i%1000 == 0:
            a, c = sess.run([accuracy, cost], feed_dict={X1: mnist.test.images, Y: mnist.test.labels})
            print("accuracy = ", a)
print("Done")

