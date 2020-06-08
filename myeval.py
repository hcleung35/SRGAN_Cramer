import os
import time
import random
import numpy as np
import scipy, multiprocessing
import tensorflow as tf
import tensorlayer as tl
from model_cam import get_G
from config import config


def myevaluate(imidx):

    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.MYVALID.lr_img_path, regx='.*.png', printable=False))

    valid_lr_imgs = tl.vis.read_images(valid_lr_img_list, path=config.MYVALID.lr_img_path, n_threads=32)

    valid_lr_img = valid_lr_imgs[imidx]
    valid_lr_img = (valid_lr_img / 127.5) - 1
    print("no. of img",  len(valid_lr_img))

    G = get_G([1, None, None, 3])
    G.load_weights(os.path.join(config.MODELS.g_model, 'g.h5'))
    G.eval()

    valid_lr_img = np.asarray(valid_lr_img, dtype=np.float32)
    valid_lr_img = valid_lr_img[np.newaxis,:,:,:]
    size = [valid_lr_img.shape[1], valid_lr_img.shape[2]]

    out = G(valid_lr_img).numpy()

    print("LR size: %s /  generated HR size: %s" % (size, out.shape))  # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
    print("[*] save images")
    tl.vis.save_image(out[0], os.path.join('valid_gen_%d.png'% imidx))
    
    out_bicu = scipy.misc.imresize(valid_lr_img[0], [size[0] * 4, size[1] * 4], interp='bicubic', mode=None)
    tl.vis.save_image(out_bicu, os.path.join('valid_bicubic_%d.png'% imidx))

    out_bicu2 = scipy.misc.imresize(valid_lr_img[0], [size[0] * 4, size[1] * 4], interp='nearest', mode=None)
    tl.vis.save_image(out_bicu2, os.path.join('valid_nearest_%d.png'% imidx))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='srgan_wgp', help='srgan_wgp, evaluate')
    parser.add_argument('imidx', help="img index", type=int , default=0)

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'myevaluate':
        myevaluate(args.imidx)
        print(args.imidx)
    else:
        raise Exception("Unknow --mode")
