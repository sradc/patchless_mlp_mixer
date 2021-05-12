import numpy as np
import tensorflow as tf

s = tf.random.normal(shape=[3, 11, 5, 7, 3])
t = tf.random.normal(shape=[7, 6])
y = tf.einsum('abcde,df->abcfe', s, t)
print(y.shape)

X = tf.random.normal(shape=[3, 11, 5, 7, 3])
w = tf.random.normal(shape=[5, 6])
y = tf.einsum('abcde,cf->abfde', X, w)
print(y.shape)

a = tf.random.uniform([3, 2])
b = tf.random.uniform([2, 3])
y = np.einsum("ij, jk -> ik", a, b)
print(y.shape)

a = tf.random.uniform([3, 2])
b = tf.random.uniform([3, 2])
y = np.einsum("ij, kj -> ik", a, b)
print(y.shape)

a = tf.random.uniform([5, 3, 2])
b = tf.random.uniform([5, 3, 2])
y = np.einsum("aij, akj -> aik", a, b)
print(y.shape)

s = tf.random.normal(shape=[3, 11, 1, 5, 3])
t = tf.random.normal(shape=[3, 1, 7, 3, 2])
e =  tf.einsum('...ij,...jk->...ik', s, t)
print(e.shape)

a = tf.random.uniform([5, 3, 2])
b = tf.random.uniform([5, 3, 2])
y = np.einsum("aij, akj -> aik", a, b)
print(y.shape)
