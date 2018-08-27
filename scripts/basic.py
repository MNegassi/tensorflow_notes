#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-

import tensorflow as tf


# To get the value of a
a = tf.add(3, 5)
with tf.Session() as sess:  # create a session, assign it to variable
    print(sess.run(a))      # Evaluate the graph

# Session allocates memory to store current value of variables (e.g a)


def subgraphs():
    """Basic tensorflow operations

    Returns
    -------

    """
    x = 2
    y = 3
    add_op = tf.add(x, y)
    mul_op = tf.multiply(x, y)
    # session doesn't evaluate useless because pow_op doesn't depend on it
    useless = tf.multiply(x, add_op)  # => saves computation
    pow_op = tf.pow(add_op, mul_op)
    with tf.Session() as sess:
        result, not_useless = sess.run([pow_op, useless])
        print(result, not_useless)

    # tf.Session.run(fetches, feed_dict=None, options=None, run_metadata=None )
    # fetches is a list of tensors whose values you want


subgraphs()


def distrib():
    """To put part of a graph on a specific CPU or GPU
    Returns
    -------
    TODO

    """
    # Create a graph
    with tf.device('/cpu:0'):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name='b')
        c = tf.multiply(a, b)

    # Creates a session with log_device_placement set to True

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    # Runs the op
    print("sess", sess.run(c))


distrib()


def graphs():
    """User created graph and default graph and Multiple Graphs
    Returns
    -------
    TODO

    """
    #
    g = tf.Graph()
    # to add operators on a Graph, set it as default
    with g.as_default():
        x = tf.add(3, 5)
    sess = tf.Session(graph=g)
    with tf.Session() as sess:
        sess.run(x)
