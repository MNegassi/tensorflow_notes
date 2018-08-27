#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-

# Operations(ops): the nodes of the graph,
#       >> describe calculations that consume and produce tensors
# Tensors: the edges in the graph.
#       >> Represent values that will flow through the graph
#       >> most tensorflow functions return tf.Tensors

# Using tf.constants makes loading graphs expensive when constants are big
# => check tensor_content in print out graph_def (stored in protobuf)
# => use constants for primitive types
# => uses variables or readers for more data that requires memory

import tensorflow as tf


def constants():
    # tf.constant(value, dtype=None, shape=None, name='Const', verify_shape=False)
    a = tf.constant([2, 2], name='a')
    b = tf.constant([[0, 1], [2, 3]], name='b')
    x = tf.multiply(a, b, name='mul')
    div = tf.div(b, a)
    np_div = tf.divide(b, a)  # =>(tf.truediv, tf.floordiv, tf.realdiv, tf.truncatediv, tf.floor_div)

    # create tensors of shape and all elements zeros
    # tf.zeros(shape, dtype=tf.float32, name=None)

    zero = tf.zeros([2, 3], tf.int32)
    zero_like = tf.zeros_like(x)  # creates zero tensor with shape/type as x
    ones_like = tf.ones_like(x)
    fill = tf.fill([2, 3], 9)
    # create sequence from (start, stop, num) of num evenly spaces
    # values in a sequence increase by stop-start / num-1 => last value is exactly stop
    seq = tf.linspace(2.0, 22.0, 5)
    rng = tf.range(5)  # rng tensor object is not iterable

    with tf.Session() as sess:
        # print("graph_def", sess.graph.as_graph_def())
        x_out, zero_out, zero_like_out, ones_like_out, fill_out = sess.run([x, zero, zero_like, ones_like, fill])
        print(sess.run([seq, rng, div, np_div]))
        print("\n x", x_out, "\nzero_like", zero_like_out, "\nzero", zero_out, "\nones_like", ones_like_out, "fill", fill_out)


constants()


def variables():
    """TODO: Docstring for variables.
    """

    # s = tf.Variable(2, name="scalar")  #=> :(
    s = tf.get_variable("scalar", initializer=tf.constant(2))  # => :)

    m = tf.get_variable("matrix", initializer=tf.constant([[0, 1], [2, 3]]))

    W = tf.get_variable("big_matrix", shape=(784, 10), initializer=tf.zeros_initializer())

    with tf.Session() as sess:
        # You have to initialize your variables => initializer is an op, you need to execute it inside context of session
        sess.run(tf.global_variables_initializer())  # initialize all variables at once
        sess.run(tf.variables_initializer([s, m]))  # initialize only subset of variables
        sess.run(W.initializer)  # initialize single variable
        print(W)
        scalar, matrix, zeros = sess.run([s, m, W])
        print("scalar", scalar, "matrix", matrix, "zeros", zeros)


def main():
    variables()


if __name__ == "__main__":
    main()

# TODO:  continue Lec 02 slide 56
