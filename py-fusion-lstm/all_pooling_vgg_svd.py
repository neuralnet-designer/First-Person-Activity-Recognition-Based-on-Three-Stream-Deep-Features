
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
    final_feature = np.zeros((video_num, 17, 512*4 ), dtype=float)
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
        #features_of_raw=np.load('../dataset/jpl/finetunned_featuremap/raw_featuremap/'+str(i+1)+'.npy')
        #features_of_raw = np.load('../dataset/jpl/finetunned_featuremap/human_diff_featuremap/' + str(i + 1) + '.npy')
        #features_of_raw = np.load('../dataset/jpl/finetunned_featuremap/diff_featuremap/' + str(i + 1) + '.npy')
        features_of_raw=np.load('../dataset/jpl/finetunned_featuremap/human_diff_featuremap_2/'+str(i+1)+'.npy')  ## this!!!
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

            channel_raw=np.zeros((512,7,7),dtype=float)
            channel_flow=np.zeros((512,7,7),dtype=float)
            #human_spatiotemporal=np.zeros((512,49),dtype=float)
            channel_optical=np.zeros((512,7,7),dtype=float)

            # flatten 512*7*7 -> 512*49
            for z in range(512): #512
                #channel_raw[z,:]=np.ravel(frame_feature_raw[z,:,:])
                #channel_flow[z,:]=np.ravel(frame_feature_flow[z,:,:])
                #channel_optical[z,:]=np.ravel(frame_feature_optical[z,:,:])

                SVD_ha=np.linalg.svd(frame_feature_raw[z,:,:])
                u_ha,s_ha,v_ha=SVD_ha
                #print(v_ha[0].shape)
                #print(frame_feature_raw[z, :, :])
                #channel_raw[z, :,:]=s_ha[0]*np.outer(u_ha.T[0],v_ha[0])
                #print(channel_raw[z,:,:])
                xa_ha=np.inner(frame_feature_raw[z,:,:],u_ha[3])
                xb_ha=np.inner(frame_feature_raw[z, :, :], v_ha[3])
                attention_ha = np.inner(xa_ha.T,xb_ha) #value
                #print(attention_ha)
                attention_ha = xa_ha.T* xb_ha  # matrix
                #print(attention_ha)
                #attention_ha=np.average(xa_ha)
                #attention_ha = np.outer(xa_ha , xb_ha)


                SVD_hm = np.linalg.svd(frame_feature_flow[z, :, :])
                u_hm, s_hm, v_hm = SVD_hm
                xa_hm = np.inner(frame_feature_flow[z, :, :], u_hm[3])
                xb_hm = np.inner(frame_feature_flow[z, :, :], v_hm[3])
                attention_hm = xa_hm.T* xb_hm
                #attention_hm = np.average(xa_hm)


                SVD_cm = np.linalg.svd(frame_feature_optical[z, :, :])
                u_cm, s_cm, v_cm = SVD_cm
                xa_cm = np.inner(frame_feature_optical[z, :, :], u_cm[3])
                xb_cm = np.inner(frame_feature_optical[z, :, :], v_cm[3])
                #attention_cm = np.inner(xa_cm, xb_cm) #valye
                attention_cm = xa_cm.T* xb_cm
                #attention_cm = np.average(xa_cm)
                #attention_cm = np.outer(xa_cm , xb_cm)
                #print(attention_cm)


                #human_camera[j, z] = np.average(attention_ha) + np.average(attention_hm) + np.average(attention_cm)+ np.max(attention_ha) + np.max(attention_hm) + np.max(attention_cm)
                #human_camera[j, z] = attention_ha+attention_cm
                #human_camera[j, z] = np.inner(xa_ha,xa_cm)
                #human_camera[j, z] = np.average(xa_ha)+np.average(xa_cm)+np.max(xa_ha)+np.max(xa_cm)
                human_camera[j, z] = np.max(attention_ha) + np.max(attention_cm) + np.max(attention_hm)

                # human_camera[j, z] = np.max(attention_ha) + np.max(attention_cm)+ np.max(attention_hm)+np.average(attention_ha) + np.average(attention_cm)+ np.average(attention_hm)




                #temp[:,0] = pca_spatiotemporal.T
            #temp[:,1] = pca_optical.T
            #pca_human_camera=pca.fit_transform(temp)

            #human_camera[j, :] = pca_spatiotemporal+pca_optical



        #SubAction-wise fusion features
        for j in range(17): #17
            Y=np.concatenate([human_camera[j,:],human_camera[j+1,:],human_camera[j+2,:],human_camera[j+3,:]]) # 512*3 dimension
            #Y = human_camera[j, :]
            Y=np.fft.fft(Y)/len(Y)

            #pca=PCA(n_components=7)
            #sub_event=pca.fit_transform(Y)

            final_feature[i,j,:]=Y




    #np.save('features/finetunned_ucf101_vgg_jpl_all_fusion_subevent_15.npy', final_feature)
    #np.save('features/finetunned_ucf101_vgg_jpl_all_fusion_subevent_14.npy', final_feature)
    #np.save('features/finetunned_ucf101_vgg_jpl_all_fusion_subevent_13.npy', final_feature)
    #np.save('features/finetunned_ucf101_vgg_jpl_all_fusion_subevent_11.npy', final_feature)
    #np.save('features/finetunned_ucf101_vgg_jpl_all_fusion_subevent_9.npy', final_feature)
    #np.save('features/finetunned_ucf101_vgg_jpl_all_fusion_subevent_16.npy', final_feature)

    np.save('features/finetunned_ucf101_vgg_jpl_all_fusion_svd_1.npy', final_feature)


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