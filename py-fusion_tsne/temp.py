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


def main():
    # video_num=179
    video_num = 84  # 84
    human_camera = np.zeros((20, 512), dtype=float)
    final_feature = np.zeros((video_num, 17, 512 * 4), dtype=float)
    coeff = np.zeros((17, 512), dtype=float)

    for i in range(video_num):
        print('video: ' + str(i + 1))

        """
        features_of_flow = np.load('../dataset/jpl/finetunned_featuremap/temp_human_flow_featuremap/' + str(i + 1) + '.npy')
        features_of_raw = np.load('../dataset/jpl/finetunned_featuremap/temp_human_raw_featuremap/' + str(i + 1) + '.npy')
        features_of_optical = np.load('../dataset/jpl/finetunned_featuremap/temp_nohuman_flow_featuremap/' + str(i + 1) + '.npy')
        """

        features_of_flow = np.load(
            '../dataset/jpl/finetunned_featuremap/human_flow_featuremap_2/' + str(i + 1) + '.npy')
        # features_of_raw=np.load('../dataset/jpl/finetunned_featuremap/raw_featuremap/'+str(i+1)+'.npy')
        # features_of_raw = np.load('../dataset/jpl/finetunned_featuremap/human_diff_featuremap/' + str(i + 1) + '.npy')
        # features_of_raw = np.load('../dataset/jpl/finetunned_featuremap/diff_featuremap/' + str(i + 1) + '.npy')         -> 92~95%
        features_of_raw = np.load(
            '../dataset/jpl/finetunned_featuremap/human_raw_featuremap_2_2/' + str(i + 1) + '.npy')
        # features_of_raw = np.load('../dataset/jpl/finetunned_featuremap/human_diff_featuremap_2/' + str(i + 1) + '.npy')  ## this!!!
        features_of_optical = np.load(
            '../dataset/jpl/finetunned_featuremap/nohuman_flow_featuremap_2/' + str(i + 1) + '.npy')

        # Channel-wise Human(appearance-motion)-camera fusion features
        for j in range(20):  # 20
            frame_feature_raw = features_of_raw[j, :, :, :]
            frame_feature_flow = features_of_flow[j, :, :, :]
            frame_feature_optical = features_of_optical[j, :, :, :]

            # flatten 512*7*7 -> 512*49
            for z in range(512):  # 512
                human_camera[j, z] = np.average(frame_feature_flow[z, :, :])

                """
                SVD_ha = np.linalg.svd(np.dot((frame_feature_raw[z, :, :] + frame_feature_optical[z, :, :]).T,
                                              (frame_feature_raw[z, :, :] + frame_feature_optical[z, :, :])))

                #SVD_ha = np.linalg.svd(frame_feature_raw[z, :, :] + frame_feature_flow[z, :, :])
                #SVD_ha = np.linalg.svd(np.dot(frame_feature_raw[z, :, :].T, frame_feature_raw[z, :, :]))
                u_ha, s_ha, v_ha = SVD_ha
                xa_ha = np.dot(frame_feature_flow[z, :, :], u_ha[2].T)
                xb_ha = np.dot(frame_feature_flow[z, :, :], v_ha[2].T)  # 1*7
                attention_ha = xa_ha * xb_ha


                SVD_hm = np.linalg.svd(np.dot((frame_feature_raw[z, :, :] + frame_feature_optical[z, :, :]).T,
                                              (frame_feature_raw[z, :, :] + frame_feature_optical[z, :, :])))
                u_hm, s_hm, v_hm = SVD_hm
                xa_hm = np.dot(frame_feature_raw[z, :, :] + frame_feature_optical[z, :, :], u_hm[2].T)
                xb_hm = np.dot(frame_feature_raw[z, :, :] + frame_feature_optical[z, :, :], v_hm[2].T)
                attention_hm = xa_hm * xb_hm


                SVD_cm = np.linalg.svd(np.dot((frame_feature_flow[z, :, :] + frame_feature_optical[z, :, :]).T,
                                              (frame_feature_flow[z, :, :] + frame_feature_optical[z, :, :])))
                u_cm, s_cm, v_cm = SVD_cm
                xa_cm = np.dot(frame_feature_flow[z, :, :] + frame_feature_optical[z, :, :], u_cm[2].T)
                xb_cm = np.dot(frame_feature_flow[z, :, :] + frame_feature_optical[z, :, :], v_cm[2].T)
                attention_cm = xa_cm * xb_cm


                human_camera[j, z] = np.average(frame_feature_raw[z, :, :] + frame_feature_flow[z, :, :]) + \
                                     np.average(frame_feature_flow[z, :, :] + frame_feature_optical[z, :, :]) + \
                                     np.average(frame_feature_raw[z, :, :] + frame_feature_optical[z, :, :])+ \
                                     np.average(frame_feature_raw[z, :, :] + frame_feature_optical[z, :,:] + frame_feature_flow[z, :, :])+np.max(attention_ha)
"""

        # SubAction-wise fusion features
        for j in range(17):  # 17
            Y = np.concatenate([human_camera[j, :], human_camera[j + 1, :], human_camera[j + 2, :],
                                human_camera[j + 3, :]])  # 512*3 dimension
            # Y = human_camera[j, :]
            Y = np.fft.fft(Y) / len(Y)
            final_feature[i, j, :] = Y

    # np.save('features/finetunned_ucf101_vgg_jpl_all_fusion_subevent_15.npy', final_feature)
    # np.save('features/finetunned_ucf101_vgg_jpl_all_fusion_subevent_14.npy', final_feature)
    # np.save('features/finetunned_ucf101_vgg_jpl_all_fusion_subevent_13.npy', final_feature)
    # np.save('features/finetunned_ucf101_vgg_jpl_all_fusion_subevent_11.npy', final_feature)
    # np.save('features/finetunned_ucf101_vgg_jpl_all_fusion_subevent_9.npy', final_feature)
    # np.save('features/finetunned_ucf101_vgg_jpl_all_fusion_subevent_16.npy', final_feature)

    # np.save('features/finetunned_ucf101_vgg_jpl_all_svd_9_xx.npy', final_feature)
    np.save('features/finetunned_ucf101_vgg_jpl_avg_humanMotion.npy', final_feature)


if __name__ == "__main__":
    main()