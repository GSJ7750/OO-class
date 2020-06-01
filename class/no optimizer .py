
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets


# In[2]:


mnist = read_data_sets('data', one_hot=True)


# In[3]:


X = tf.placeholder(tf.float32, shape=[None, 784])
Y = tf.placeholder(tf.float32, shape=[None, 10])


# In[4]:


def sigmoid(x):
    sigmoidX = tf.div(tf.constant(1.0), #1.0/(1+e^-x)
                   tf.add ( tf.constant(1.0), tf.exp(tf.negative(x)) )
                   )
    return sigmoidX


# In[5]:


def softmax(x):
    X = tf.exp(x) / tf.reduce_sum(tf.exp(x), axis=0)
    return X


# In[18]:


W1 = tf.Variable(tf.random_normal([784, 200], stddev=0.1))
b1 = tf.Variable(tf.random_normal([200], stddev=0.1))
z1 = tf.add(tf.matmul(X, W1), b1) #layer1 logit
L1 = sigmoid(tf.add(tf.matmul(X, W1), b1)) #sigmoid, layer1

W2 = tf.Variable(tf.random_normal([200, 100], stddev=0.1))
b2 = tf.Variable(tf.random_normal([100], stddev=0.1))
z2 = tf.add(tf.matmul(L1, W2), b2) #layer logit2
L2 = sigmoid(tf.add(tf.matmul(L1, W2), b2)) #layer2

W3 = tf.Variable(tf.random_normal([100, 10], stddev=0.1))
b3 = tf.Variable(tf.random_normal([10], stddev=0.1))
logits = tf.add(tf.matmul(L2, W3), b3)# output logit
H = softmax(logits)# Hypothesis
#순전파


# In[19]:


def sigmoidprime(x):
    sigmoidPX = tf.multiply(sigmoid(x), 
                          tf.subtract( tf.constant(1.0), sigmoid(x) )
                         )
    return sigmoidPX


# In[20]:


diff = tf.subtract(H, Y)#예측값-실제값(오차)


# In[21]:


d_z3 = tf.multiply(diff, sigmoidprime(logits))#오차 x logits, 결과값에 로짓이 미치는 영향, 체인룰
d_b3 = d_z3
d_W3 = tf.matmul(tf.transpose(L2), d_z3)


# In[22]:


d_L2 = tf.matmul(d_z3, tf.transpose(W3))
d_z2 = tf.multiply(d_L2, sigmoidprime(z2))
d_b2 = d_z2
d_W2 = tf.matmul(tf.transpose(L1), d_z2)


# In[23]:


d_L1 = tf.matmul(d_z2, tf.transpose(W2))
d_z1 = tf.multiply(d_L1, sigmoidprime(z1))
d_b1 = d_z1
d_W1 = tf.matmul(tf.transpose(X), d_z1)
#역전파


# In[24]:


eta = tf.constant(0.5)


# In[25]:


step = [
    tf.assign(W1, tf.subtract(W1, tf.multiply(eta, d_W1)))
  , tf.assign(b1, tf.subtract(b1, tf.multiply(eta, tf.reduce_mean(d_b1, axis=0))))
    
  , tf.assign(W2, tf.subtract(W2, tf.multiply(eta, d_W2)))
  , tf.assign(b2, tf.subtract(b2, tf.multiply(eta, tf.reduce_mean(d_b2, axis=0))))
    
  , tf.assign(W3, tf.subtract(W3, tf.multiply(eta, d_W3)))
  , tf.assign(b3, tf.subtract(b3, tf.multiply(eta, tf.reduce_mean(d_b3, axis=0))))
]


# In[26]:


is_correct = tf.equal(tf.argmax(H, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))


# In[27]:


sess = tf.Session()
sess.run(tf.global_variables_initializer())


# In[28]:


for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(10)
    sess.run(step, feed_dict = {X: batch_xs, Y : batch_ys})
    if i % 1000 == 0:
        a = sess.run(accuracy, feed_dict ={X: mnist.test.images,Y : mnist.test.labels})
        print(a)


# In[29]:


for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(10)
    sess.run(step, feed_dict = {X: batch_xs, Y : batch_ys})
    if i % 1000 == 0:
        h, a = sess.run([H, accuracy], feed_dict ={X: mnist.test.images,Y : mnist.test.labels})
        print(a,"\n", h)

