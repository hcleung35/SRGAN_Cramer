import cv2
import PIL
import os
import time
import random
import numpy as np
import scipy, multiprocessing
import tensorflow as tf
import tensorlayer as tl
import moviepy.editor as mp
from config import config
from model_cam import get_G
from PIL import Image 
from os.path import isfile, join


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()


def super_res(video):   

    pathIn = video
    file_type = video[-4:]
    pathOut = video[:-4] + "_sr" + file_type
    pathOut_audio = video[:-4] + "_sr_audio" + file_type
    cap = cv2.VideoCapture(pathIn)
    frame_array = []

    G = get_G([1, None, None, 3])
    G.load_weights(os.path.join(config.MODELS.g_model, 'g.h5'))
    G.eval()

    while not cap.isOpened():
        cap = cv2.VideoCapture(pathIn)
        cv2.waitKey(1000)
        print ("Wait for the header")

    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    while True:
        flag, frame = cap.read()
        if flag:
            lr_img = (frame / 127.5) - 1
            lr_img = np.asarray(lr_img, dtype=np.float32)
            lr_img = lr_img[np.newaxis,:,:,:]

            out = G(lr_img).numpy()
            img = out[0]
            size = (img.shape[1], img.shape[0])
            img = (img + 1) * 127.5
            scaled_img = img.astype(np.uint8)
            frame_array.append(scaled_img)

            pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            printProgressBar(pos_frame + 1, total_frame, prefix = 'Progress:', suffix = 'Complete', length = 50)
        else:

            cap.set(cv.CV_CAP_PROP_POS_FRAMES, pos_frame-1)
            print ("frame is not ready")
     
            cv2.waitKey(1000)

        if cv2.waitKey(10) == 27:
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            break

    fps = cap.get(cv2.CAP_PROP_FPS)
    if file_type == '.mp4':
        out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    if file_type == '.avi':
        out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()

    video_org = mp.VideoFileClip(pathIn)
    audio_org = video_org.audio
    video = mp.VideoFileClip(pathOut)

    final = video.set_audio(audio_org)
    final.write_videofile(pathOut_audio, fps = fps, codec='libx264')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("video", type=str)

    args = parser.parse_args()
    t = args.video
    super_res(args.video)
