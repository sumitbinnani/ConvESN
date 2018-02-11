"""
Example - "a01_s01_e01_skeleton.txt"
>>> 20 action types
>>> 10 subjects
>>> each subject performs each action 2 or 3 times
"""
import os
import csv
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import cPickle
import random


def load_MSRA3D_P4(filepath, params):
    print "Loading Skeleton data sets MSRA3D...."
    flag_normalize = True
    Samples = []
    Labels=[]
    action_type = params[0]
    subjects = params[1]
    times = params[2]

    nums_skeleton = 20
    max_frames = 0

    for a in action_type:
        for s in subjects:
            for t in times:
                # load each sample
                fileName = filepath + "/" + 'a%02i_s%02i_e%02i_skeleton3D.txt' % (a, s, t)
                print "Processing file a%02i_s%02i_e%02i_skeleton3D.txt .........." % (a, s, t)
                flag = os.path.exists(fileName)
                if flag == True:
                    fr = open(fileName)
                    frameMat = []
                    jointMat = []
                    for line in fr.readlines():
                        lineArr = line.strip().split(' ')
                        jointMat.append([float(lineArr[0]), float(lineArr[1]), float(lineArr[2])])
                    nums_frame = np.shape(jointMat)[0] / nums_skeleton
                    for i in range(nums_frame):
                        mat_frame = []
                        for j in range(nums_skeleton):
                            index = i * nums_skeleton + j
                            mat_frame.extend(jointMat[index])
                        frameMat.append(mat_frame)
                    frameMat = np.array(frameMat)

                    n_frames, n_features = frameMat.shape
                    temp = frameMat
                    frameMat_temp = np.zeros((n_frames-4, n_features))

                    # Savitzky-Golay smoothing filter
                    for i in range(2, n_frames - 2):
                        frameMat_temp[i-2,:] = (-3*temp[i-2,:]+12*temp[i-1,:]+17*temp[i,:]+12*temp[i+1,:]-3*temp[i+2,:]) / 35
                    frameMat = frameMat_temp
                    n_frames, n_features = frameMat.shape

                    if n_frames > max_frames:
                        max_frames = n_frames

                    # Centralization
                    temp = frameMat
                    frameMat_temp = np.zeros((n_frames, n_features))
                    for i in range(n_frames):
                        Origin = (temp[i, 12:15] + temp[i, 15:18] + temp[i, 18:21]) / 3
                        for j in range(nums_skeleton):
                            index = 3*j
                            frameMat_temp[i, index] = temp[i, index] - Origin[0]
                            frameMat_temp[i, index + 1] = temp[i, index + 1] - Origin[1]
                            frameMat_temp[i, index + 2] = temp[i, index + 2] - Origin[2]

                    # Normalization
                    frameMat = frameMat_temp
                    if flag_normalize:
                        for j in range(n_features):
                            frameMat[:, j] = frameMat[:, j] - mean(frameMat[:, j])

                    Samples.append(frameMat)
                    Labels.extend([a-1])
    Samples = np.array(Samples)
    flag_splite5parts = True
    if flag_splite5parts:

        Samples_left_hand =[]
        Samples_right_hand =[]
        Samples_left_leg =[]
        Samples_right_leg =[]
        Samples_central_trunk =[]
        nums_samples = len(Samples)

        for i in range(nums_samples):
            sample = Samples[i]
            nums_frames = sample.shape[0]

            Frames_left_hand =[]
            Frames_right_hand =[]
            Frames_left_leg =[]
            Frames_right_leg =[]
            Frames_central_trunk =[]

            for j in range(nums_frames):

                left_hand_joint = []
                for k in [1, 8, 10, 12]:
                    left_hand_joint.extend(list(Samples[i][j][(k-1)*3:k*3]))

                right_hand_joint = []
                for k in [2, 9, 11, 13]:
                    right_hand_joint.extend(list(Samples[i][j][(k-1)*3:k*3]))

                left_leg_joint = []
                for k in [5, 14, 16, 18]:
                    left_leg_joint.extend(list(Samples[i][j][(k-1)*3:k*3]))

                right_leg_joint = []
                for k in [6, 15, 17, 19]:
                    right_leg_joint.extend(list(Samples[i][j][(k-1)*3:k*3]))

                central_trunk_joint = []
                for k in [20, 3, 4, 7]:
                    central_trunk_joint.extend(list(Samples[i][j][(k-1)*3:k*3]))

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

    results = np.array(Samples_left_hand), np.array(Samples_right_hand), np.array(Samples_left_leg), \
              np.array(Samples_right_leg), np.array(Samples_central_trunk), np.array(Labels), max_frames

    return results

if __name__ == '__main__':
    filepath = "./data/MSRAction3D_real_world"
    """
        In Protocol 4, the most widely adopted, cross-subject validation with subjects 1,3,5,7, and 9 for training, the others for test (HS-V
        protocol: Half subjects to test, the rest for training) is adopted.
        """
    AS1_a = [2, 3, 5, 6, 10, 13, 18, 20]
    AS2_a = [1, 4, 7, 8, 9, 11, 12, 14]
    AS3_a = [6, 14, 15, 16, 17, 18, 19, 20]

    suject_train = [1, 3, 5, 7, 9]
    suject_test = [2, 4, 6, 8, 10]

    AS1_train_params = [AS1_a, suject_train, [1, 2, 3]]
    AS2_train_params = [AS2_a, suject_train, [1, 2, 3]]
    AS3_train_params = [AS3_a, suject_train, [1, 2, 3]]
    AS1_test_params = [AS1_a, suject_test, [1, 2, 3]]
    AS2_test_params = [AS2_a, suject_test, [1, 2, 3]]
    AS3_test_params = [AS3_a, suject_test, [1, 2, 3]]

    max_frames = [0 for i in range(6)]

    [AS1_train_left_hand, AS1_train_right_hand, AS1_train_left_leg, AS1_train_right_leg, AS1_train_central_trunk,AS1_train_Labels, max_frames[0]] = load_MSRA3D_P4(filepath, AS1_train_params)
    [AS2_train_left_hand, AS2_train_right_hand, AS2_train_left_leg, AS2_train_right_leg, AS2_train_central_trunk, AS2_train_Labels, max_frames[1]] = load_MSRA3D_P4(filepath, AS2_train_params)
    [AS3_train_left_hand, AS3_train_right_hand, AS3_train_left_leg, AS3_train_right_leg, AS3_train_central_trunk, AS3_train_Labels, max_frames[2]] = load_MSRA3D_P4(filepath, AS3_train_params)

    [AS1_test_left_hand, AS1_test_right_hand, AS1_test_left_leg, AS1_test_right_leg, AS1_test_central_trunk, AS1_test_Labels, max_frames[3]] = load_MSRA3D_P4(filepath, AS1_test_params)
    [AS2_test_left_hand, AS2_test_right_hand, AS2_test_left_leg, AS2_test_right_leg, AS2_test_central_trunk, AS2_test_Labels, max_frames[4]] = load_MSRA3D_P4(filepath, AS2_test_params)
    [AS3_test_left_hand, AS3_test_right_hand, AS3_test_left_leg, AS3_test_right_leg, AS3_test_central_trunk, AS3_test_Labels, max_frames[5]] = load_MSRA3D_P4(filepath, AS3_test_params)

    flag_created = True
    if flag_created:
        cPickle.dump([AS1_train_left_hand, AS1_train_right_hand, AS1_train_left_leg, AS1_train_right_leg, AS1_train_central_trunk, AS1_train_Labels, max_frames[0]], open("./data/DataBackUp/MSRAction3D_real_world_P4_Split_AS1_train.p", "wb"))
        cPickle.dump([AS2_train_left_hand, AS2_train_right_hand, AS2_train_left_leg, AS2_train_right_leg, AS2_train_central_trunk, AS2_train_Labels, max_frames[1]], open("./data/DataBackUp/MSRAction3D_real_world_P4_Split_AS2_train.p", "wb"))
        cPickle.dump([AS3_train_left_hand, AS3_train_right_hand, AS3_train_left_leg, AS3_train_right_leg, AS3_train_central_trunk, AS3_train_Labels, max_frames[2]], open("./data/DataBackUp/MSRAction3D_real_world_P4_Split_AS3_train.p", "wb"))
        cPickle.dump([AS1_test_left_hand, AS1_test_right_hand, AS1_test_left_leg, AS1_test_right_leg, AS1_test_central_trunk,AS1_test_Labels, max_frames[3]], open("./data/DataBackUp/MSRAction3D_real_world_P4_Split_AS1_test.p", "wb"))
        cPickle.dump([AS2_test_left_hand, AS2_test_right_hand, AS2_test_left_leg, AS2_test_right_leg, AS2_test_central_trunk,AS2_test_Labels, max_frames[4]], open("./data/DataBackUp/MSRAction3D_real_world_P4_Split_AS2_test.p", "wb"))
        cPickle.dump([AS3_test_left_hand, AS3_test_right_hand, AS3_test_left_leg, AS3_test_right_leg, AS3_test_central_trunk,AS3_test_Labels, max_frames[5]], open("./data/DataBackUp/MSRAction3D_real_world_P4_Split_AS3_test.p", "wb"))
        print "Dataset created!"







