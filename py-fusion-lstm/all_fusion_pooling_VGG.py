
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
from torch.autograd import Variable

from sklearn.decomposition import PCA


def main():


    video_num=89

    #avg_feature=np.zeros((video_num,20,2048), dtype=float)
    fused_feaures = np.zeros((video_num, 18, 512), dtype=float)

    #avg_feature = np.zeros((video_num, 20, 2048), dtype=float)

    for i in range(1):
        print('video: '+str(i+1))

        channel_wise_product = np.zeros((20, 512, 7, 7), dtype=float)

        features_of_flow = np.load('../dataset/jpl/finetunned_featuremap/human_flow_featuremap/' + str(i + 1) + '.npy')
        features_of_raw=np.load('../dataset/jpl/finetunned_featuremap/human_raw_featuremap/'+str(i+1)+'.npy')
        features_of_optical = np.load('../dataset/jpl/finetunned_featuremap/nohuman_flow_featuremap/' + str(i + 1) + '.npy')
        """
        features_of_flow = np.load('../dataset/utk/featuremaps/utk_raw_featuremap/' + str(i + 1) + '.npy')
        features_of_raw = np.load('../dataset/utk/featuremaps/utk_flow_featuremap/' + str(i + 1) + '.npy')
        features_of_optical = np.load('../dataset/utk/featuremaps/utk_nohuman_flow_featuremap/' + str(i + 1) + '.npy')
        """

        for j in range(20):
            frame_feature_raw= features_of_raw[j,:,:,:]
            frame_feature_flow = features_of_flow[j, :, :, :]
            frame_feature_optical = features_of_optical[j, :, :, :]
            #print(len(frame_feature_raw)) # 512


            for z in range(512):
                #print('start')
                channel_raw = frame_feature_raw[z,:,:]
                #print(channel_raw)
                channel_flow=frame_feature_flow[z,:,:]
                #print(channel_flow)
                channel_optical=frame_feature_optical[z,:,:]

                #channel_wise_product[z,:,:]=np.dot(channel_raw,channel_flow)
                channel_wise_product[j,z, :, :] = channel_raw+channel_flow # consider two feature map (spatio-temporal)
                #print(channel_wise_product[j,z,:,:])
                #print(channel_optical)
                channel_wise_product[j,z, :, :]=np.inner( channel_wise_product[j,z,:,:],channel_optical) # consider interaction of two feature map (human-camera interaction)
                #print(channel_wise_product[j,z,:,:])

            #frame wise - pooling (512*7*7 -> 4096)
            #channel_wise ~

        #3 frame wise (sub-event) - pooling
        temp_channel= np.zeros((512,3,7,7), dtype=float)
        for f in range(18):
            print(f)
            for c in range(512):
                temp_channel[c,0,:,:] = channel_wise_product[f,c,:,:]
                temp_channel[c,1,:,:] = channel_wise_product[f+1,c, :, :]
                temp_channel[c,2,:,:] = channel_wise_product[f+2,c, :, :]

            tensor_temp_channel = torch.from_numpy(temp_channel[ :, :, :])
            #print(tensor_temp_channel)
            tensor_temp_channel = tensor_temp_channel.float()
            #tensor_temp_channel = Variable(tensor_temp_channel)

            pooling3d = nn.MaxPool3d((3, 3, 3), stride=(3,1,1))  # have to test with no stride (stride=(2,2,2)
            output = pooling3d(tensor_temp_channel)  # 512*1*3*3
            output1=output[:,0,:,:] # 512*5*5
            #print(output1)

            linear = nn.Linear(512 *5 * 5, 512)

            linear_output = linear(output1)
            print(linear_output)
            fused_feaures[i, f, :] = linear_output



    """
        for k in range(18):
            tensor_temp_channel = torch.from_numpy(temp_channel[k,:,:,:,:])
            tensor_temp_channel=tensor_temp_channel.float()
            tensor_temp_channel = Variable(tensor_temp_channel)
            pooling3d = nn.MaxPool3d((3, 3, 3), stride = (3,1,1))  # have to test with no stride (stride=(2,2,2)

            output = pooling3d(tensor_temp_channel)  # 512*3*3
            print(output)
            linear1 = nn.Linear(512 *1* 5 * 5, 512)

            linear_output = linear1(output)
            print(linear_output)

            fused_feaures[i, k, :] = linear_output

"""



                #np.save('features/ucf101_vgg_utk_all_fusion_2048_2.npy',avg_feature)




if __name__=="__main__":
    main()