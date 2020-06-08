from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()

## Adam
config.TRAIN.batch_size = 8 # [16] use 8 if your GPU memory is small, and use [2, 4] in tl.vis.save_images / use 16 for faster training
config.TRAIN.lr_init = 1e-4
config.TRAIN.beta1 = 0.5
config.TRAIN.beta2 = 0.9

## initialize G
config.TRAIN.n_epoch_init = 50


## adversarial learning (SRGAN)
config.TRAIN.n_epoch = 500
config.TRAIN.lr_decay = 0.5
config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 5)

## train set location
#config.TRAIN.hr_img_path = 'DIV2K_train_HR/'
#config.TRAIN.lr_img_path = 'DIV2K_train_LR_bicubic/X4/'
config.TRAIN.hr_img_path = 'srgan_w/DIV2K_train_HR/'
config.TRAIN.lr_img_path = 'srgan_w/DIV2K_train_LR_bicubic/X4/'

config.MYVALID = edict()
## test set location
config.MYVALID.lr_img_path = 'myval/'
config.MODELS = edict()
config.MODELS.g_model = 'models/'

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
