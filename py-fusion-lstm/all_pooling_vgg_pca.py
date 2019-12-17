
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


    #video_num=179
    video_num = 84 #84
    human_camera=np.zeros((20,512), dtype=float)
    final_feature = np.zeros((video_num, 17, 512 * 4), dtype=float)
    coeff = np.zeros((17, 512), dtype=float)

    for i in range(video_num):
        print('video: '+str(i+1))


        """
        features_of_flow = np.load('../dataset/jpl/finetunned_featuremap/temp_human_flow_featuremap/' + str(i + 1) + '.npy')
        features_of_raw = np.load('../dataset/jpl/finetunned_featuremap/temp_human_raw_featuremap/' + str(i + 1) + '.npy')
        features_of_optical = np.load(
            '../dataset/jpl/finetunned_featuremap/temp_nohuman_flow_featuremap/' + str(i + 1) + '.npy')


        """
        features_of_flow = np.load('../dataset/jpl/finetunned_featuremap/human_flow_featuremap_2/' + str(i + 1) + '.npy')
        features_of_raw=np.load('../dataset/jpl/finetunned_featuremap/raw_featuremap/'+str(i+1)+'.npy')
        #features_of_raw = np.load('../dataset/jpl/finetunned_featuremap/human_diff_featuremap/' + str(i + 1) + '.npy')
        #features_of_raw = np.load('../dataset/jpl/finetunned_featuremap/diff_featuremap/' + str(i + 1) + '.npy')
        #features_of_raw=np.load('../dataset/jpl/finetunned_featuremap/human_diff_featuremap_2/'+str(i+1)+'.npy')  ## this!!!
        features_of_optical = np.load('../dataset/jpl/finetunned_featuremap/nohuman_flow_featuremap_2/' + str(i + 1) + '.npy')

        """
        features_of_raw = np.load('../dataset/utk/featuremaps/utk_human_raw_featuremap/' + str(i + 1) + '.npy')
        #features_of_raw = np.load('../dataset/utk/featuremaps/utk_human_diff_featuremap/' + str(i + 1) + '.npy')
        #features_of_raw = np.load('../dataset/utk/featuremaps/utk_raw_diff_featuremap/' + str(i + 1) + '.npy')
        features_of_flow = np.load('../dataset/utk/featuremaps/utk_human_flow_featuremap/' + str(i + 1) + '.npy')
        features_of_optical = np.load('../dataset/utk/featuremaps/utk_nohuman_flow_featuremap/' + str(i + 1) + '.npy')
        """

        # Channel-wise Human(appearance-motion)-camera fusion features
        for j in range(20): #20
            frame_feature_raw= features_of_raw[j,:,:,:]
            frame_feature_flow = features_of_flow[j,:,:,:]
            frame_feature_optical = features_of_optical[j,:,:,:]

            channel_raw=np.zeros((512,49),dtype=float)
            channel_flow=np.zeros((512,49),dtype=float)
            #human_spatiotemporal=np.zeros((512,49),dtype=float)
            channel_optical=np.zeros((512,49),dtype=float)


            # flatten 512*7*7 -> 512*49
            for z in range(512): #512
                channel_raw[z,:]=np.ravel(frame_feature_raw[z,:,:])
                channel_flow[z,:]=np.ravel(frame_feature_flow[z,:,:])
                channel_optical[z,:]=np.ravel(frame_feature_optical[z,:,:])


            #calculate principal component 49->1
            pca =PCA(n_components=1)
            pca_raw = np.ravel(pca.fit_transform(channel_raw))
            pca_flow = np.ravel(pca.fit_transform(channel_flow))
            pca_optical = np.ravel(pca.fit_transform(channel_optical))

            temp = np.zeros((512, 2), dtype=float)

            temp[:,0]=pca_raw.T
            temp[:,1]=pca_flow.T
            pca_spatiotemporal = np.ravel(pca.fit_transform(temp))

            #temp[:,0] = pca_spatiotemporal.T
            #temp[:,1] = pca_optical.T
            #pca_human_camera=pca.fit_transform(temp)

            human_camera[j, :] = pca_spatiotemporal+pca_optical


            """
            #correlation of human and camera-ego motion
            lst2=[np.ravel(pca_raw), np.ravel(pca_flow),np.ravel(pca_optical)]
            df2=pd.DataFrame(lst2).T
            corr2=df2.corr(method='pearson')
            print(corr2)
            
            #human_spatiotemporal=(channel_max_raw+channel_max_flow)+corr2[0][1]
            #human_camera[j,:] = (human_spatiotemporal + channel_max_optical) * corr2[0][1]
            #human_camera[j, :] = (human_spatiotemporal + channel_max_optical) + corr2[1][2]
            #print(human_camera[j,:])

            
            if not math.isnan(corr[0][1]) :
                print(index)
                index=index+1
                print(corr[0][1])
                print("-----------------")
            """




        #SubAction-wise fusion features
        for j in range(17): #17
            Y=np.concatenate([human_camera[j,:],human_camera[j+1,:],human_camera[j+2,:],human_camera[j+3,:]]) # 512*3 dimension
            #Y = np.concatenate([human_camera[j, :], human_camera[j + 1, :], human_camera[j + 2, :]])
            sub_event=np.fft.fft(Y)/len(Y)

            #pca=PCA(n_components=7)
            #sub_event=pca.fit_transform(Y)

            final_feature[i,j,:]=sub_event





    #np.save('features/finetunned_ucf101_vgg_jpl_all_fusion_subevent_15.npy', final_feature)
    #np.save('features/finetunned_ucf101_vgg_jpl_all_fusion_subevent_14.npy', final_feature)
    #np.save('features/finetunned_ucf101_vgg_jpl_all_fusion_subevent_13.npy', final_feature)
    #np.save('features/finetunned_ucf101_vgg_jpl_all_fusion_subevent_11.npy', final_feature)
    #np.save('features/finetunned_ucf101_vgg_jpl_all_fusion_subevent_9.npy', final_feature)
    #np.save('features/finetunned_ucf101_vgg_jpl_all_fusion_subevent_16.npy', final_feature)

    np.save('features/finetunned_ucf101_vgg_jpl_all_fusion_pca_1.npy', final_feature)


    #np.save('features/finetunned_ucf101_vgg_utk_all_fusion_subevent_20.npy',final_feature)  # original model + sub-event(3frames fft)
    #np.save('features/finetunned_ucf101_vgg_utk_all_fusion_subevent_13.npy',final_feature)  # original model + sub-event(3frames fft)
    #np.save('features/finetunned_ucf101_vgg_utk_all_fusion_subevent_14.npy',final_feature)  # original model + sub-event(3frames fft)
    #np.save('features/finetunned_ucf101_vgg_utk_all_fusion_subevent_12.npy',final_feature) #original model + sub-event(3frames fft)
    #np.save('features/finetunned_ucf101_vgg_utk_all_fusion_subevent_11.npy', final_feature) #model_12 + add crop
    #np.save('features/finetunned_ucf101_vgg_utk_all_fusion_subevent_9.npy',final_feature)
    #np.save('features/finetunned_ucf101_vgg_utk_all_fusion_subevent_16.npy',final_feature)  # original model + sub-event(3frames fft)
    #np.save('features/finetunned_ucf101_vgg_utk_all_fusion_subevent_17.npy',final_feature)  # original model + sub-event(3frames fft)
    #np.save('features/finetunned_ucf101_vgg_utk_all_fusion_subevent_19.npy',final_feature)  # original model + sub-event(3frames fft)



if __name__=="__main__":
    main()