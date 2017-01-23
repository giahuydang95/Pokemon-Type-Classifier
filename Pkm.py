#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 12:28:08 2017

@author: huy
"""
from __future__ import print_function

import numpy as np
import math
import pandas as pd
import time
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import KFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    os.chdir('/home/huy/Desktop')
    path = os.getcwd()
    
    pkm = pd.read_csv(path + '/Pokemon.csv')    
    useless = ['#', 'Legendary', 'Generation']
    pkm = pkm.drop(useless, axis = 1)

    features = pkm.columns
    features = [x.encode('utf-8') for x in features]
    features.remove('Type 1')
    features.remove('Type 2')
    stat = features

    mm = StandardScaler()
    X = pkm[stat].values
    types = (pkm['Type 1'].values)
    count = 0
    d = {}
    for t in types:
        if t in d:
            continue
        else:
            d[t] = count
            count += 1
            
    reversed_d = dict(zip(d.values(), d.keys()))
            
    for i in d.keys():
        vec = np.zeros(len(np.unique(types)))
        vec[d[i]] = 1
        d[i] = vec

    types_new = np.zeros((len(types), len(np.unique(types))), dtype = np.float32)
    for i in range(len(types)):
        types_new[i,:] = d[types[i]]


    X_train, X_test, y_train, y_test = train_test_split(X, types_new, test_size = 0.125)
    temp = X_test #For latter use
    X_train = np.delete(X_train, [0,3], axis = 1)
    X_test = np.delete(X_test, [0,3],axis=1)
    stat.remove('Name')
    stat.remove('Attack')
    
    batch_size = 100
    num_labels = len(np.unique(types))
    
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.2)
    
    X_train = (mm.fit_transform(X_train)).astype(np.float32)
    X_test = mm.fit_transform(X_test).astype(np.float32)
    X_valid = mm.fit_transform(X_valid).astype(np.float32)
    
    graph = tf.Graph()
#    num_hidden_nodes_1 = 216
    num_hidden_nodes = 512
    dropout = 0.7
    
    with graph.as_default():
        
        tf_train_dataset = tf.placeholder(tf.float32, shape = (batch_size,len(stat)))
        tf_train_labels = tf.placeholder(tf.float32, shape = (batch_size, 18 ))
    
        tf_valid_dataset = tf.constant(X_valid)
        tf_test_dataset = tf.constant(X_test)
        
        weights_1 = tf.Variable(
                                tf.truncated_normal([len(stat), num_hidden_nodes], stddev = .04))
        weights_2 = tf.Variable(
                                tf.truncated_normal([num_hidden_nodes, num_labels], stddev = .04))
        
        biases_1 = tf.Variable(
                               tf.zeros([num_hidden_nodes]))
        biases_2 = tf.Variable(
                               tf.zeros([num_labels]))
        
        layer_1 = tf.nn.relu(tf.matmul(tf_train_dataset, weights_1) + biases_1)
        layer_1 = tf.nn.dropout(layer_1, dropout)

        logits = tf.matmul(layer_1, weights_2) + biases_2

        C = 5e-7
        R = tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2) + tf.nn.l2_loss(biases_1) + tf.nn.l2_loss(biases_2)

        loss = tf.reduce_mean(
                          tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels) + C*R)
        
        global_steps = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(0.04, global_steps, decay_steps=1000, decay_rate=0.2)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_steps)
        
        train_prediction = tf.nn.softmax(logits)
        
        layer1_valid = tf.nn.relu(tf.matmul(tf_valid_dataset, weights_1) + biases_1)
        valid_prediction = tf.nn.softmax(tf.matmul(layer1_valid, weights_2) + biases_2)
        
        layer_1_test = tf.nn.relu(tf.matmul(tf_test_dataset, weights_1) + biases_1)
        test_prediction = tf.nn.softmax(tf.matmul(layer_1_test, weights_2) + biases_2)

    num_steps = 12000

    def accuracy(predictions, labels):
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
              / predictions.shape[0])
            
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print('Initialized')
        for step in range(num_steps):
            offset = (step*batch_size) % (X_train.shape[0] - batch_size)
        
            batch_data = X_train[offset:(offset + batch_size),:]
            batch_labels = y_train[offset:(offset + batch_size), :]
        
            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}

            _, l, predictions = session.run(
                                        [optimizer, loss, train_prediction], feed_dict=feed_dict)
            if step % 500 == 0:
                print("Minibatch loss at step %d: %f" % (step, l))
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                print("Validation accuracy: %.1f%%" % accuracy(
                                                           valid_prediction.eval(), y_valid))
        print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), y_test))
 #       pred = (test_prediction.eval())
        
    
#    pred = map(int,np.argmax(pred, axis = 1))
 #   for i in range(len(temp)):
 #       print('For {} with stats: HP = {}, Attack = {}, Defense = {}, Sp.Attack = {}, Sp.Defense= {}, Speed = {}'.format(temp[i,0], temp[i,2],
 #                                                     temp[i,3], temp[i,4], temp[i,5], temp[i,6], temp[i,7]))
 #       print('The Pokémon should be type: {}\n'.format(reversed_d[pred[i]]))
        