#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-

# Operations(ops): the nodes of the graph,
#       >> describe calculations that consume and produce tensors
# Tensors: the edges in the graph.
#       >> Represent values that will flow through the graph
#       >> most tensorflow functions return tf.Tensors

import tensorflow as tf


def constants():
    # tf.constant(value, dtype=None, shape=None, name='Const', verify_shape=False)
    a = tf.constant([2, 2], name='a')
    b = tf.constant([[0, 1], [2, 3]], name='b')
    x = tf.multiply(a, b, name='mul')

    # create tensors of shape and all elements zeros
    # tf.zeros(shape, dtype=tf.float32, name=None)

    zero = tf.zeros([2, 3], tf.int32)

    with tf.Session() as sess:
        x, zero = sess.run([x, zero])
        print(x, "\n", zero)

constants()

# TODO: Continue Lec 02 slides from page 19
