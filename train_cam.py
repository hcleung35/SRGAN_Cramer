#! /usr/bin/python
# -*- coding: utf8 -*-

import os
import time
import random
import datetime
import numpy as np
import scipy, multiprocessing
import tensorflow as tf
import tensorlayer as tl
from model_cam import get_G, get_D
from config import config

###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = config.TRAIN.batch_size  # use 8 if your GPU memory is small, and change [4, 4] in tl.vis.save_images to [2, 4]
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
beta2 = config.TRAIN.beta2
## initialize G
n_epoch_init = config.TRAIN.n_epoch_init
## adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every
shuffle_buffer_size = 128
# ni = int(np.sqrt(batch_size))

# create folders to save result images and trained models
save_dir = "samples"
tl.files.exists_or_mkdir(save_dir)
checkpoint_dir = "models"
tl.files.exists_or_mkdir(checkpoint_dir)

def get_train_data():
    # load dataset
    train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))#[0:20]


    ## If your machine have enough memory, please pre-load the entire train set.
    train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
        
    length = len(train_hr_imgs)
    print("lenght:" , length)
    train_hr_imgs_len = (length - batch_size)

    # dataset API and augmentation
    def generator_train():
        for i in range(train_hr_imgs_len):
            yield train_hr_imgs[i]
    def generator_valid():
        for i in range(batch_size):
            yield train_hr_imgs[train_hr_imgs_len+i]

    def _map_fn_train(img):
        hr_patch = tf.image.random_crop(img, [192, 192, 3])
        hr_patch = hr_patch / (255. / 2.)
        hr_patch = hr_patch - 1.
        hr_patch = tf.image.random_flip_left_right(hr_patch)
        lr_patch = tf.image.resize(hr_patch, size=[48, 48])
        return lr_patch, hr_patch
    train_ds = tf.data.Dataset.from_generator(generator_train, output_types=(tf.float32))
    train_ds = train_ds.map(_map_fn_train, num_parallel_calls=multiprocessing.cpu_count())

    train_ds = train_ds.shuffle(shuffle_buffer_size)
    train_ds = train_ds.prefetch(buffer_size=2)
    train_ds = train_ds.batch(batch_size)


    valid_ds = tf.data.Dataset.from_generator(generator_valid, output_types=(tf.float32))
    valid_ds = valid_ds.map(_map_fn_train, num_parallel_calls=multiprocessing.cpu_count())
    valid_ds = valid_ds.shuffle(shuffle_buffer_size)
    valid_ds = valid_ds.prefetch(buffer_size=2)
    valid_ds = valid_ds.batch(batch_size)

    return train_ds, valid_ds, train_hr_imgs_len

def train():
    G = get_G((batch_size, 48, 48, 3))
    D = get_D((batch_size, 192, 192, 3))
    VGG = tl.models.vgg19(pretrained=True, end_with='pool4', mode='static')
    epsilon2 = tf.random.uniform([], 0.0, 1.0)

    lr_v = tf.Variable(lr_init)
    g_optimizer_init = tf.optimizers.Adam(lr_v, beta_1=beta1, beta_2=beta2)
    g_optimizer = tf.optimizers.Adam(lr_v, beta_1=beta1, beta_2=beta2)
    d_optimizer = tf.optimizers.Adam(lr_v, beta_1=beta1, beta_2=beta2)

    G.train()
    D.train()
    VGG.train()
    scale = 10.0

    train_ds, valid_ds, no_img ,= get_train_data()

 
    ## adversarial learning (G, D)
    n_step_epoch = round(no_img // batch_size)
    for epoch in range(n_epoch):
        for step, (lr_patchs, hr_patchs) in enumerate(train_ds):
            for i, (lr_patchs_valid, _) in enumerate(valid_ds):
                #tl.vis.save_images(lr_patchs_valid.numpy(), [2,4], os.path.join(save_dir, 'valid_{}.png'.format(step)))
                print()
            if lr_patchs.shape[0] != batch_size: # if the remaining data in this epoch < batch_size
                break
            step_time = time.time()
            with tf.GradientTape(persistent=True) as tape:

                # train discrimator
                net_g = G(lr_patchs)
                net_g2 = G(lr_patchs_valid) #True?
                yy1 = D(net_g)
                yy2 = D(net_g2,  training = False)

                xx1 = D(hr_patchs)

                d_loss_pre  = (tf.norm(yy1 - yy2, axis = 1) - tf.norm(yy1, axis = 1)) - (tf.norm(xx1 - yy2, axis = 1) - tf.norm(xx1, axis = 1))     
                x_hat = epsilon2 * xx1 + (1 - epsilon2) * yy1
                with tf.GradientTape(persistent=True) as tape2:
                    tape2.watch(x_hat)
                    ddx_pre = ((tf.norm(x_hat - yy2, axis = 1) - tf.norm(x_hat, axis = 1)))
                ddx_pre2 = tape2.gradient(ddx_pre, x_hat)

                ddx = tf.square(tf.norm(ddx_pre2, axis=1) - 1.0) * scale

                d_loss = (d_loss_pre + ddx)
            grad_d = tape.gradient(d_loss, D.trainable_weights)

            d_optimizer.apply_gradients(zip(grad_d, D.trainable_weights))
            

            with tf.GradientTape(persistent=True) as tape3:
                #train generator
                xx1 = D(hr_patchs)
                yy2_pre = G(lr_patchs_valid, training = False)
                yy2 = D(yy2_pre)

                net_g = G(lr_patchs)
                yy1 = D(net_g)

                feature_fake = VGG((net_g+1)/2.) # the pre-trained VGG uses the input range of [0, 1]
                feature_real = VGG((hr_patchs+1)/2.)

                
                g_gan_loss =  ((tf.norm(xx1 - yy2, axis = 1) - tf.norm(xx1, axis = 1)) - (tf.norm(yy1 - yy2, axis = 1) - tf.norm(yy1, axis = 1))) 
                #g_gan_loss = (tf.norm(xx1 - yy1, axis = 1) + tf.norm(xx1 - yy2, axis = 1) - tf.norm(yy1 - yy2, axis = 1))
                mse_loss = tl.cost.mean_squared_error(net_g, hr_patchs, is_mean=True)
                vgg_loss = 2e-6 * tl.cost.mean_squared_error(feature_fake, feature_real, is_mean=True)
                g_loss = mse_loss + vgg_loss + 1e-4 *g_gan_loss
            grad_g = tape3.gradient(g_loss, G.trainable_weights)

            g_optimizer.apply_gradients(zip(grad_g, G.trainable_weights))

            d_loss_mean = tf.reduce_mean(d_loss, axis = 0)
            g_loss_mean = tf.reduce_mean(1e-4 *g_gan_loss, axis = 0)
            

            
            print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, d_loss: {:.3f}, g_loss(mse:{:.3f}, vgg:{:.3f}, adv:{:.3f})".format(
                epoch, n_epoch, step, n_step_epoch, time.time() - step_time, d_loss_mean, mse_loss, vgg_loss, g_loss_mean))

        # update the learning rate

        if epoch != 0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay**(epoch // decay_every)
            lr_v.assign(lr_init * new_lr_decay)
            log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
            print(log)

        if (epoch != 0) and (epoch % 10 == 0):
            lr_patchs3 = tf.image.resize(lr_patchs, size=[192, 192])
            temp_patch4 = tf.concat((lr_patchs3[0:4], net_g[0:4]), 0)
            temp_patch5 = tf.concat((lr_patchs3[4:8], net_g[4:8]), 0)
            temp_patch6 = tf.concat((temp_patch4, temp_patch5), 0)

            tl.vis.save_images(temp_patch6.numpy(), [4, 4], os.path.join(save_dir, 'train_g_{}.png'.format(epoch)))
            G.save_weights(os.path.join(checkpoint_dir, 'g.h5'))
            D.save_weights(os.path.join(checkpoint_dir, 'd.h5'))




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='srgan_cam', help='srgan, evaluate')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'srgan_cam':
        train()

    else:
        raise Exception("Unknow --mode")
