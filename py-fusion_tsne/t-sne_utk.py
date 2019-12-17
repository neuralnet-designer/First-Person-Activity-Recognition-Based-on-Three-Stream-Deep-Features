import os
import sys
import collections
import numpy as np
import math
import random
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import cv2
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def main():

    video_num=179
    #video_num =84
    human_camera = np.zeros((20, 512), dtype=float)
    #final_feature = np.zeros((video_num, 17, 512 * 4), dtype=float)
    #final_feature = np.zeros((video_num*17, 512 * 4), dtype=float)
    final_feature_test = np.zeros(((video_num/2) * 17, 512 * 4), dtype=float)
    label_test=[]

    val_file = "jpl_test2.txt"
    f_val = open(val_file, "r")
    val_list = f_val.readlines()

    col=-1

    for line in val_list:
        col=col+1

        line_info=line.split(" ")
        video_num=line_info[0] #video number
        video_label=int(line_info[1]) # video label

        print('video: ' + str(video_num))

        # features_of_raw=np.load('../dataset/jpl/finetunned_featuremap/raw_featuremap/'+str(i+1)+'.npy')
        # features_of_raw = np.load('../dataset/jpl/finetunned_featuremap/human_diff_featuremap/' + str(i + 1) + '.npy')
        # features_of_raw = np.load('../dataset/jpl/finetunned_featuremap/diff_featuremap/' + str(i + 1) + '.npy')         -> 92~95%
        #features_of_raw = np.load('../dataset/jpl/finetunned_featuremap/human_raw_featuremap_2/' + str(video_num) + '.npy')
        #features_of_raw = np.load('../dataset/jpl/finetunned_featuremap/human_diff_featuremap_2/' + str(video_num) + '.npy')  ## this!!!
        #features_of_flow = np.load('../dataset/jpl/finetunned_featuremap/human_flow_featuremap_2/' + str(video_num) + '.npy')
        #features_of_optical = np.load('../dataset/jpl/finetunned_featuremap/nohuman_flow_featuremap_2/' + str(video_num) + '.npy')

        features_of_raw = np.load('../dataset/utk/featuremaps/utk_human_raw_featuremap/' + str(i + 1) + '.npy')
        # features_of_raw = np.load('../dataset/utk/featuremaps/utk_human_diff_featuremap/' + str(i + 1) + '.npy')
        # features_of_raw = np.load('../dataset/utk/featuremaps/utk_raw_diff_featuremap/' + str(i + 1) + '.npy')
        features_of_flow = np.load('../dataset/utk/featuremaps/utk_human_flow_featuremap/' + str(i + 1) + '.npy')
        features_of_optical = np.load('../dataset/utk/featuremaps/utk_nohuman_flow_featuremap/' + str(i + 1) + '.npy')

        for j in range(20):  # 20

            frame_feature_raw = features_of_raw[j, :, :, :]
            frame_feature_flow = features_of_flow[j, :, :, :]
            frame_feature_optical = features_of_optical[j, :, :, :]

            # flatten 512*7*7 -> 512*49
            for z in range(512):  # 512


                SVD_ha = np.linalg.svd(np.dot((frame_feature_raw[z, :, :]+frame_feature_flow[z, :, :]).T, frame_feature_raw[z, :, :]+frame_feature_flow[z, :, :]))
                u_ha, s_ha, v_ha = SVD_ha
                xa_ha = np.dot(frame_feature_raw[z, :, :], u_ha[2].T)
                xb_ha = np.dot(frame_feature_raw[z, :, :], v_ha[2].T)  # 1*7
                attention_ha = xa_ha*xb_ha

                SVD_hm = np.linalg.svd(np.dot(frame_feature_flow[z, :, :].T, frame_feature_flow[z, :, :]))
                u_hm, s_hm, v_hm = SVD_hm
                xa_hm = np.dot(frame_feature_flow[z, :, :], u_hm[2].T)
                xb_hm = np.dot(frame_feature_flow[z, :, :], v_hm[2].T)
                attention_hm = xa_hm * xb_hm

                SVD_cm = np.linalg.svd(np.dot(frame_feature_optical[z, :, :].T, frame_feature_optical[z, :, :]))
                u_cm, s_cm, v_cm = SVD_cm
                xa_cm = np.dot(frame_feature_flow[z, :, :] + frame_feature_optical[z, :, :], u_cm[2].T)
                xb_cm = np.dot(frame_feature_flow[z, :, :] + frame_feature_optical[z, :, :], v_cm[2].T)
                attention_cm = xa_cm * xb_cm

                human_camera[j, z] = np.average(frame_feature_raw[z, :, :])

                temp=np.max(frame_feature_raw[z, :, :] + frame_feature_flow[z, :, :]) + \
                       np.max(frame_feature_raw[z, :, :]+ frame_feature_optical[z, :, :]) + \
                       np.max(frame_feature_optical[z, :, :] + frame_feature_flow[z, :, :])
                        #np.max(frame_feature_optical[z, :, :] + frame_feature_flow[z, :, :]+frame_feature_raw[z, :, :])

                #human_camera[j, z] =np.average(frame_feature_optical[z, :, :] + frame_feature_flow[z, :, :])



        # SubAction-wise fusion features
        for j in range(17):  # 17
            Y = np.concatenate([human_camera[j, :], human_camera[j + 1, :], human_camera[j + 2, :], human_camera[j + 3, :]])  # 512*3 dimension
            # Y = human_camera[j, :]
            Y = np.fft.fft(Y) / len(Y)
            final_feature_test[col*17+j, :] = Y
            label_test.append(video_label)

    cols=[]
    for l in label_test:
        if l == 1:
            cols.append('red')   #hand shake
        elif l == 2:
            cols.append('green') #hug
        elif l == 3:
            cols.append('blue')  #pet
        elif l == 4:
            cols.append('pink')  #wave
        elif l == 5:
            cols.append('purple')#point
        elif l == 6:
            cols.append('yellow')#punch
        elif l == 7:
            cols.append('orange')#throw
        else:
            cols.append('black')

    label = ('hand shake', 'hug', 'pet', 'wave', 'point', 'punch', 'throw')
    color = ('blue','green','red','pink','purple','yellow','orange')

    model=TSNE(learning_rate=100)
    transformed=model.fit_transform(final_feature_test)
    xs = transformed[:, 0]
    ys = transformed[:, 1]
    plt.scatter(xs,ys,c=cols)
    plt.show()



if __name__ == "__main__":
    main()