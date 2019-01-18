# -*- coding: utf-8 -*-
import os
import re
import pickle
import numpy as np
import tensorflow as tf

 
def attention_layer(inputs, attention_size=None, return_alphas=False):
    if isinstance(inputs, tuple):  # 防止B-LSTM前向与反向输出没有合并
        inputs = tf.concat(inputs, 2)
  
    if attention_size is None:
        hidden_size = inputs.shape[-1]
    else:
        hidden_size = attention_size

    uit = tf.layers.dense(inputs=inputs, units=hidden_size, activation=tf.nn.tanh)
    a = tf.exp(uit)
    a /= tf.cast(tf.reduce_sum(a, axis=1, keep_dims=True), "float64")
    weighted_input = inputs * a
    outputs = tf.reduce_sum(weighted_input, axis=1)
    # print(output.shape)
    return outputs  # [batch_size,dimensions]


def train_rnn_model(embedding_matrix, x_train, y_train, bs, epoch):
    """
    made by bidirectional LSTM, attention layer and dense layer

    :param embedding_matrix: ndarray, word embedding
    :param x_train: ndarray, train set
    :param y_train: ndarray, train label
    :param bs: int, batch size
    :param epoch: int, repeat training times
    :return: int, if successful get 0 else get 1
    """
    try:
        with tf.Graph().as_default():
            x = tf.placeholder(dtype=tf.int32, shape=[None, 100], name='input')
            y = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='label')
            batch_size = tf.placeholder(dtype=tf.int32, shape=[], name='batch')

            # 得到词向量形式的输入
            embedding = tf.Variable(embedding_matrix, name='Embeddings')
            input_embedding = tf.nn.embedding_lookup(embedding, x)  # Tensor

            # 搭建 B-LSTM 结构
            stacked_fw_rnn = []
            stacked_bw_rnn = []
            for _ in range(1):  # 前向
                lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=256, state_is_tuple=True)
                stacked_fw_rnn.append(lstm_fw_cell)
            for _ in range(1):  # 反向
                lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=256, state_is_tuple=True)
                stacked_bw_rnn.append(lstm_bw_cell)

            mlstm_fw_cell = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_fw_rnn, state_is_tuple=True)
            mlstm_bw_cell = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_bw_rnn, state_is_tuple=True)

            init_lstm_fw_state = mlstm_fw_cell.zero_state(batch_size, dtype=tf.float64)  # 用全零来初始化state
            init_lstm_bw_state = mlstm_bw_cell.zero_state(batch_size, dtype=tf.float64)

            outputs, state = tf.nn.bidirectional_dynamic_rnn(mlstm_fw_cell, mlstm_bw_cell, inputs=input_embedding,
                                                             initial_state_fw=init_lstm_fw_state,
                                                             initial_state_bw=init_lstm_bw_state)
            h_state = tf.concat(outputs, 2)

            att_state = attention_layer(h_state)  # 注意力层
            dense_1 = tf.layers.dense(inputs=att_state, units=128, activation=tf.nn.relu)  # 全连接层
            y_hat = tf.layers.dense(inputs=dense_1, units=1, activation=tf.nn.sigmoid)
            prediction = tf.round(y_hat, name='predict')  # tf.round 返回类型必须为float

            num = tf.constant(100.0, dtype=tf.float64)
            probability = tf.divide(tf.round(tf.multiply(y_hat, num)), num, name='prob')  # 保留两位小数

            # 设置模型训练
            y_label = tf.cast(y, "float64")
            cost = -tf.reduce_mean(y_label * tf.log(y_hat) + (1-y_label) * tf.log(1-y_hat))  # 对数损失函数
            train = tf.train.AdamOptimizer(1e-3).minimize(cost)  # 设计训练方式

            # 评估
            correct_prediction = tf.equal(prediction, y_label)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float64"), name='accuracy')

            # 创建saver
            saver = tf.train.Saver()

            # 初始化
            init = tf.global_variables_initializer()

            # 开启会话
            print('begin this session')
            with tf.Session() as sess:
                sess.run(init)
                batch_num = int(x_train.shape[0] / bs)
                for e in range(epoch):
                    for k in range(batch_num):
                        batch = [x_train[k*bs:(k+1)*bs], y_train[k*bs:(k+1)*bs]]
                        if (k + 1) % 10 == 0:
                            acc = sess.run(accuracy, feed_dict={x: batch[0], y: batch[1], batch_size: bs})
                            print("Epoch %d, step %d, training accuracy %g" % (e+1, k+1, acc))

                        sess.run(train, feed_dict={x: batch[0], y: batch[1], batch_size: bs})

                saver.save(sess, './model/blstm_att')

        return 0

    except Exception as err:
        print(err)
        return 1


def test_model_without_def(x_test, y_test, meta, bs=1):
    with tf.Graph().as_default() as graph:
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph('./model/'+meta)
            saver.restore(sess, tf.train.latest_checkpoint('./model/'))

            x = graph.get_tensor_by_name('input:0')
            y = graph.get_tensor_by_name('label:0')

            batch_size = graph.get_tensor_by_name('batch:0')
            accuracy = graph.get_tensor_by_name('accuracy:0')

            batch = [x_test[:bs], y_test[:bs]]
            result = sess.run(accuracy, feed_dict={x: batch[0], y: batch[1], batch_size: bs})

    return result


def predict(x_test, meta, bs=1):
    with tf.Graph().as_default() as graph:
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph('./model/'+meta)
            saver.restore(sess, tf.train.latest_checkpoint('./model/'))

            x = graph.get_tensor_by_name('input:0')
            batch_size = graph.get_tensor_by_name('batch:0')

            probability = graph.get_tensor_by_name('prob:0')
            prediction = graph.get_tensor_by_name('predict:0')

            bool_prediction = tf.cast(prediction, dtype=tf.int32)  # add a new tensor in this function
            batch = x_test[:bs]
            b = sess.run(bool_prediction, feed_dict={x: batch, batch_size: bs})
            p = sess.run(probability, feed_dict={x: batch, batch_size: bs})

    return b, p

