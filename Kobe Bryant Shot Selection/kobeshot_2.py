#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 19:47:19 2018

@author: z

using CNN
"""

#kobe_dataset
import pandas as pd
import tensorflow as tf

x_data = pd.read_csv('kobe_data.csv',usecols=['loc_x','loc_y','shot_distance',
'shot_zone_area','playoffs','period','minutes_remaining','seconds_remaining',
'shot_type'])

x_data = x_data.values
y_data = pd.read_csv('kobe_data.csv',usecols=['t1','t2','t3','t4','t5','t6'])
y_data = y_data.values

x_test = x_data[30000:30600,:]
y_test = y_data[30000:30600,:]

X = tf.placeholder(tf.float32)

Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([9, 9], -1., 1.))

W2 = tf.Variable(tf.random_uniform([9, 6], -1., 1.))

b1 = tf.Variable(tf.zeros([9]))

b2 = tf.Variable(tf.zeros([6]))

L1 = tf.add(tf.matmul(X, W1), b1)

L1 = tf.nn.relu(L1)

model = tf.add(tf.matmul(L1, W2), b2)

cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model))


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(30000):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})

    if (step + 1) % 100 == 0:
        print(step + 1, sess.run(cost, feed_dict={X: x_data, Y: y_data}))

prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)

#print('Predicted value:', sess.run(prediction, feed_dict={X: x_test}))
#print('Actual value:', sess.run(target, feed_dict={Y: y_test}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('accuracy: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_test, Y: y_test}))

