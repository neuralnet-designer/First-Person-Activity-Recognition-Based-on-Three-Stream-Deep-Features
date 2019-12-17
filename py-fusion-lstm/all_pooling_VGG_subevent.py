
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
    final_feature=np.zeros((video_num,18,512*3), dtype=float)
    spatio_temporal=np.zeros((512*3,7,7), dtype=float)
    human_camera = np.zeros((512 * 3, 49, 49), dtype=float)
    #avg_feature = np.zeros((video_num, 20, 2048), dtype=float)

    for i in range(video_num):
        print('video: '+str(i+1))

        """
        features_of_flow = np.load('../dataset/jpl/finetunned_featuremap/human_flow_featuremap/' + str(i + 1) + '.npy')
        features_of_raw=np.load('../dataset/jpl/finetunned_featuremap/human_raw_featuremap/'+str(i+1)+'.npy')
        features_of_optical = np.load('../dataset/jpl/finetunned_featuremap/nohuman_flow_featuremap/' + str(i + 1) + '.npy')
        """

        features_of_flow = np.load('../dataset/utk/featuremaps/utk_raw_featuremap/' + str(i + 1) + '.npy')
        features_of_raw = np.load('../dataset/utk/featuremaps/utk_flow_featuremap/' + str(i + 1) + '.npy')
        features_of_optical = np.load('../dataset/utk/featuremaps/utk_nohuman_flow_featuremap/' + str(i + 1) + '.npy')


        for j in range(18):
            frame_feature_raw= np.concatenate([features_of_raw[j,:,:,:],features_of_raw[j+1,:,:,:],features_of_raw[j+2,:,:,:]])
            frame_feature_flow = np.concatenate([features_of_flow[j,:,:,:],features_of_flow[j+1,:,:,:],features_of_flow[j+2,:,:,:]])
            frame_feature_optical = np.concatenate([features_of_optical[j,:,:,:],features_of_optical[j+1,:,:,:],features_of_optical[j+2,:,:,:]])

            for z in range(512*3):
                channel_raw=frame_feature_raw[z, :, :]
                channel_flow=frame_feature_flow[z,:,:]
                channel_optical=frame_feature_optical[z,:,:]

                for x in range(7):
                    for y in range(7):
                        spatio_temporal[z,x,y]=(channel_raw[x,y]+channel_flow[x,y])/2


                human_camera[z,:,:] = np.outer(spatio_temporal[z,:,:],channel_optical)

                final_feature[i, j, z] = np.max(human_camera)

                """
                max_raw = np.max(frame_feature_raw[z, :, :])
                max_flow=np.max(frame_feature_flow[z,:,:])
                max_optical = np.max(frame_feature_optical[z,:,:])

                spatio_temporal = max_raw+max_flow
                human_camera=spatio_temporal+max_optical
                """



                """ 1024
                avg_raw=np.average(frame_feature_flow[z,:,:])
                avg_flow=np.average(frame_feature_raw[z,:,:])
                #avg_optical = np.average(frame_feature_optical[z, :, :])
                avg_optical = np.max(frame_feature_optical[z, :, :])

                #using average of flow and raw feature map
                avg_spatiotemporal=(avg_raw+avg_flow)/2
                avg_feature[i,j,z]=avg_spatiotemporal

                #using average of optical flow feature map
                avg_feature[i,j,z+512]=avg_optical
                #avg_feature[i, j, z + 1024] = avg_optical
                #avg_feature[i, j, z + 1536] = max_optical

                """



    np.save('features/finetunned_ucf101_vgg_utk_all_fusion_subevent_6.npy',final_feature)




if __name__=="__main__":
    main()