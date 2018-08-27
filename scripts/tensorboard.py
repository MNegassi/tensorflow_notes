#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


a = tf.constant(2, name='a')  # to name variables explicitly, use name
b = tf.constant(3, name='b')
x = tf.add(a, b, name='add')

# create summary writer after graph definition and before running session
writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
with tf.Session() as sess:
    print(sess.run(x))  # >> 5
writer.close()

# tensorboard --logdir="./any_folder_you_want" --port 6006
# To check on visualization on tensorboard: Open browser and go to http://localhost:6006/
