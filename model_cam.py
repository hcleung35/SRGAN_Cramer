#! /usr/bin/python
# -*- coding: utf8 -*-

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import (Input, Conv2d, BatchNorm1d, BatchNorm2d, Elementwise, SubpixelConv2d, Flatten, Dense)
from tensorlayer.models import Model

def get_G(input_shape):
    w_init = tf.random_normal_initializer(stddev=0.02)
    g_init = tf.random_normal_initializer(1., 0.02)

    nin = Input(input_shape)
    n = Conv2d(64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init)(nin)
    temp = n

    # B residual blocks
    for i in range(16):
        nn = Conv2d(64, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=None)(n)
        nn = BatchNorm2d(act=tf.nn.relu, gamma_init=g_init)(nn)
        nn = Conv2d(64, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=None)(nn)
        nn = BatchNorm2d(gamma_init=g_init)(nn)
        nn = Elementwise(tf.add)([n, nn])
        n = nn

    n = Conv2d(64, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(gamma_init=g_init)(n)
    n = Elementwise(tf.add)([n, temp])
    # B residual blacks end

    n = Conv2d(256, (3, 3), (1, 1), padding='SAME', W_init=w_init)(n)
    n = SubpixelConv2d(scale=2, n_out_channels=None, act=tf.nn.relu)(n)


    n = Conv2d(256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init)(n)
    n = SubpixelConv2d(scale=2, n_out_channels=None, act=tf.nn.relu)(n)


    nn = Conv2d(3, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init)(n)
    G = Model(inputs=nin, outputs=nn, name="generator")
    return G

def get_D(input_shape):
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    df_dim = 64
    lrelu = lambda x: tl.act.lrelu(x, 0.2)

    nin = Input(input_shape)
    n = Conv2d(df_dim, (4, 4), (2, 2), act=lrelu, padding='SAME', W_init=w_init)(nin)

    n = Conv2d(df_dim * 2, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 4, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 8, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 16, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 32, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 16, (1, 1), (1, 1), padding='SAME', W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 8, (1, 1), (1, 1), padding='SAME', W_init=w_init, b_init=None)(n)
    nn = BatchNorm2d(gamma_init=gamma_init)(n)

    n = Conv2d(df_dim * 2, (1, 1), (1, 1), padding='SAME', W_init=w_init, b_init=None)(nn)
    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 2, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 8, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(gamma_init=gamma_init)(n)
    n = Elementwise(combine_fn=tf.add, act=lrelu)([n, nn])

    n = Flatten()(n)
    n1 = Dense(n_units=1000 ,W_init=w_init)(n)
    no = BatchNorm1d(act = tf.nn.tanh, gamma_init=gamma_init)(n1)
    D = Model(inputs=nin, outputs=no, name="discriminator")
    return D


