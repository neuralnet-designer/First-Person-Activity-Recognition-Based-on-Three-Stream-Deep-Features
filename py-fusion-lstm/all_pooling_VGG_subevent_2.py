
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



def main():


    video_num=179
    #video_num = 84
    final_feature=np.zeros((video_num,17,512*4), dtype=float)
    human_camera=np.zeros((20,512), dtype=float)



    for i in range(video_num):
        print('video: '+str(i+1))

        """
        features_of_flow = np.load('../dataset/jpl/finetunned_featuremap/temp_human_flow_featuremap/' + str(i + 1) + '.npy')
        features_of_raw = np.load('../dataset/jpl/finetunned_featuremap/temp_human_raw_featuremap/' + str(i + 1) + '.npy')
        features_of_optical = np.load(
            '../dataset/jpl/finetunned_featuremap/temp_nohuman_flow_featuremap/' + str(i + 1) + '.npy')
        """
        
        features_of_flow = np.load('../dataset/jpl/finetunned_featuremap/human_flow_featuremap/' + str(i + 1) + '.npy')
        #features_of_raw=np.load('../dataset/jpl/finetunned_featuremap/raw_featuremap/'+str(i+1)+'.npy')
        features_of_raw = np.load('../dataset/jpl/finetunned_featuremap/human_diff_featuremap/' + str(i + 1) + '.npy')
        #features_of_raw = np.load('../dataset/jpl/finetunned_featuremap/diff_featuremap/' + str(i + 1) + '.npy')
        #features_of_raw=np.load('../dataset/jpl/finetunned_featuremap/human_raw_featuremap/'+str(i+1)+'.npy')
        features_of_optical = np.load('../dataset/jpl/finetunned_featuremap/nohuman_flow_featuremap/' + str(i + 1) + '.npy')
        """


        #features_of_raw = np.load('../dataset/utk/featuremaps/utk_human_raw_featuremap_add/' + str(i + 1) + '.npy')
        features_of_raw = np.load('../dataset/utk/featuremaps/utk_human_diff_featuremap/' + str(i + 1) + '.npy')
        #features_of_raw = np.load('../dataset/utk/featuremaps/utk_raw_diff_featuremap/' + str(i + 1) + '.npy')
        features_of_flow = np.load('../dataset/utk/featuremaps/utk_human_flow_featuremap/' + str(i + 1) + '.npy')
        features_of_optical = np.load('../dataset/utk/featuremaps/utk_nohuman_flow_featuremap/' + str(i + 1) + '.npy')
        """

        # Channel-wise Human(appearance-motion)-camera fusion features
        for j in range(20):
            frame_feature_raw= features_of_raw[j,:,:,:]
            frame_feature_flow = features_of_flow[j,:,:,:]
            frame_feature_optical = features_of_optical[j,:,:,:]

            # channel_wise sum of appearance and motion activation maps
            for z in range(512):
                channel_raw = frame_feature_raw[z, :, :]
                channel_flow = frame_feature_flow[z, :, :]
                channel_optical = frame_feature_optical[z, :, :]
                fusion = np.sum(channel_raw + channel_flow)+np.average(channel_optical)
                human_camera[j, z] = np.sum(fusion)  # 512 dimension

                """
                channel_raw=frame_feature_raw[z, :, :]
                channel_flow=frame_feature_flow[z,:,:]
                channel_optical=frame_feature_optical[z,:,:]
                fusion=np.sum(channel_raw+channel_flow+channel_optical)
                human_camera[j,z] = np.sum(fusion) # 512 dimension
                """


        #SubAction-wise fusion features
        for j in range(17):
            Y=np.concatenate([human_camera[j,:],human_camera[j+1,:],human_camera[j+2,:],human_camera[j+3,:]]) # 512*3 dimension
            sub_event=np.fft.fft(Y)/len(Y)
            final_feature[i,j,:]=sub_event



    #np.save('features/finetunned_ucf101_vgg_jpl_all_fusion_subevent_15.npy', final_feature)
    #np.save('features/finetunned_ucf101_vgg_jpl_all_fusion_subevent_14.npy', final_feature)
    #np.save('features/finetunned_ucf101_vgg_jpl_all_fusion_subevent_13.npy', final_feature)
    #np.save('features/finetunned_ucf101_vgg_jpl_all_fusion_subevent_11.npy', final_feature)
    #np.save('features/finetunned_ucf101_vgg_jpl_all_fusion_subevent_9.npy', final_feature)

    np.save('features/finetunned_ucf101_vgg_utk_all_fusion_subevent_13.npy',final_feature)  # original model + sub-event(3frames fft)
    #np.save('features/finetunned_ucf101_vgg_utk_all_fusion_subevent_14.npy',final_feature)  # original model + sub-event(3frames fft)
    #np.save('features/finetunned_ucf101_vgg_utk_all_fusion_subevent_12.npy',final_feature) #original model + sub-event(3frames fft)
    #np.save('features/finetunned_ucf101_vgg_utk_all_fusion_subevent_11.npy', final_feature) #model_12 + add crop
    #np.save('features/finetunned_ucf101_vgg_utk_all_fusion_subevent_9.npy',final_feature)


if __name__=="__main__":
    main()