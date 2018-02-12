import os
import csv
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import cPickle
import random


def Load_Florence3D(fileName):
    nums_skeleton = 15
    flag_normalize = True
    Labels = []
    Samples = []
    Subjects = []
    video_id = []
    subject_id = []
    action_id = []
    frame = []
    fr = open(fileName)
    still_maxframes = 0

    for line in fr.readlines():
        lineArr = line.strip().split(' ')
        video_id.extend([float(lineArr[0])])
        subject_id.extend([float(lineArr[1])])
        action_id.extend([float(lineArr[2])])
        frame_temp = []
        for i in range(3, len(lineArr)):
            frame_temp.extend([float(lineArr[i])])
        frame.append(frame_temp)

    frame_total_num = len(frame)
    video_id, video_id_index = np.unique(video_id, return_index=True)
    video_id_index = list(video_id_index)
    video_id_index.extend([frame_total_num])


    sample_total_num = len(video_id)

    for i in range(sample_total_num):
        action = []
        for j in range(video_id_index[i], video_id_index[i + 1]):
            action.append(frame[j])
        frameMat = action
        frameMat = np.array(frameMat)

        # Truncation
        n_frames, n_features = frameMat.shape
        if n_frames > 100:
            continue

        temp = frameMat

        # Savitzky-Golay smoothing filter
        if (n_frames >= 100):
            frameMat_temp = np.zeros((n_frames - 4, n_features))
            for p in range(2, n_frames - 2):
                frameMat_temp[p - 2, :] = (
                                              -3 * temp[p - 2, :] + 12 * temp[p - 1, :] + 17 * temp[p,
                                                                                               :] + 12 * temp[
                                                                                                         p + 1,
                                                                                                         :] - 3 * temp[
                                                                                                                  p + 2,
                                                                                                                  :]) / 35
            frameMat = frameMat_temp
        n_frames, n_features = frameMat.shape

        # Centralization
        temp = frameMat
        frameMat_temp = np.zeros((n_frames, n_features))
        for p in range(n_frames):
            Origin = (temp[p, 6:9] + temp[p, 27:30] + temp[p, 36:39]) / 3
            for j in range(nums_skeleton):
                index = 3 * j
                frameMat_temp[p, index] = temp[p, index] - Origin[0]
                frameMat_temp[p, index + 1] = temp[p, index + 1] - Origin[1]
                frameMat_temp[p, index + 2] = temp[p, index + 2] - Origin[2]


        # Normalization
        frameMat = frameMat_temp

        if flag_normalize:
            for j in range(n_features):
                frameMat[:, j] = frameMat[:, j] - mean(frameMat[:, j])
                frameMat[:, j] = 0.01 * frameMat[:, j]
                # frameMat[:, j] = frameMat[:, j] /std(frameMat[:, j])
                # frameMat[:, j] = np.tanh(20 * frameMat[:, j])
        Samples.append(frameMat)

    for j in range(sample_total_num):
        Labels.extend([action_id[video_id_index[j]] - 1])
    for j in range(sample_total_num):
        Subjects.extend([subject_id[video_id_index[j]]])

    Samples_Data = range(10)
    uni_Subjects, Subjects_index = np.unique(Subjects, return_index=True)
    Subjects_index = list(Subjects_index)
    Subjects_index.extend([len(Labels)])

    for Sub_i in range(len(uni_Subjects)):
        Samples_Data_temp = []
        Labels_temp = []
        max_frames = 0
        for Sample_j in range(Subjects_index[Sub_i], Subjects_index[Sub_i + 1]):
            Samples_Data_temp.append(Samples[Sample_j])
            n_frames = len(Samples[Sample_j])
            Labels_temp.extend([Labels[Sample_j]])

            # Update max_frame
            if n_frames > max_frames:
                max_frames = n_frames

        flag_splite5parts = True
        if flag_splite5parts:

            Samples_left_hand = []
            Samples_right_hand = []
            Samples_left_leg = []
            Samples_right_leg = []
            Samples_central_trunk = []

            nums_samples = len(Samples_Data_temp)

            for i in range(nums_samples):
                sample = Samples_Data_temp[i]
                nums_frames = sample.shape[0]

                Frames_left_hand = []
                Frames_right_hand = []
                Frames_left_leg = []
                Frames_right_leg = []
                Frames_central_trunk = []

                for j in range(nums_frames):

                    left_hand_joint = []
                    for k in [4, 5, 6]:
                        left_hand_joint.extend(list(Samples_Data_temp[i][j][(k - 1) * 3:k * 3]))

                    right_hand_joint = []
                    for k in [7, 8, 9]:
                        right_hand_joint.extend(list(Samples_Data_temp[i][j][(k - 1) * 3:k * 3]))

                    left_leg_joint = []
                    for k in [10, 11, 12]:
                        left_leg_joint.extend(list(Samples_Data_temp[i][j][(k - 1) * 3:k * 3]))

                    right_leg_joint = []
                    for k in [13, 14, 15]:
                        right_leg_joint.extend(list(Samples_Data_temp[i][j][(k - 1) * 3:k * 3]))

                    central_trunk_joint = []
                    for k in [1, 2, 3]:
                        central_trunk_joint.extend(list(Samples_Data_temp[i][j][(k - 1) * 3:k * 3]))

                    Frames_left_hand.append(left_hand_joint)
                    Frames_right_hand.append(right_hand_joint)
                    Frames_left_leg.append(left_leg_joint)
                    Frames_right_leg.append(right_leg_joint)
                    Frames_central_trunk.append(central_trunk_joint)

                Samples_left_hand.append(Frames_left_hand)
                Samples_right_hand.append(Frames_right_hand)
                Samples_left_leg.append(Frames_left_leg)
                Samples_right_leg.append(Frames_right_leg)
                Samples_central_trunk.append(Frames_central_trunk)
        if max_frames>still_maxframes:
            still_maxframes = max_frames
        Samples_Data[Sub_i] = [np.array(Samples_left_hand), np.array(Samples_right_hand), np.array(Samples_left_leg), \
                               np.array(Samples_right_leg), np.array(Samples_central_trunk), np.array(Labels_temp),
                               np.array(still_maxframes)]
    return Samples_Data


if __name__ == '__main__':
    fileName = './data/Florence3D/world_coordinates.txt'
    Samples_Data = Load_Florence3D(fileName)
    flag_created = True
    if flag_created:
        cPickle.dump(Samples_Data, open("./data/DataBackUp/Florence3D_DataSet.p", "wb"))
    print "Load finished!"




