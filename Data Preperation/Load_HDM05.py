import scipy.io as sio
import numpy as np
import os
import csv
from numpy import *
import cPickle
import random


def load_HDM05 (filepath):
    action_name = ['cartwheelLHandStart1Reps','cartwheelLHandStart2Reps','cartwheelRHandStart1Reps','clap1Reps','clap5Reps','clapAboveHead1Reps','clapAboveHead5Reps','depositFloorR','depositHighR','depositLowR','depositMiddleR','elbowToKnee1RepsLelbowStart','elbowToKnee1RepsRelbowStart','elbowToKnee3RepsLelbowStart','elbowToKnee3RepsRelbowStart','grabFloorR','grabHighR','grabLowR','grabMiddleR','hitRHandHead','hopBothLegs1hops','hopBothLegs2hops','hopBothLegs3hops','hopLLeg1hops','hopLLeg2hops','hopLLeg3hops','hopRLeg1hops','hopRLeg2hops','hopRLeg3hops','jogLeftCircle4StepsRstart','jogLeftCircle6StepsRstart','jogOnPlaceStartAir2StepsLStart','jogOnPlaceStartAir2StepsRStart','jogOnPlaceStartAir4StepsLStart','jogOnPlaceStartFloor2StepsRStart','jogOnPlaceStartFloor4StepsRStart','jogRightCircle4StepsLstart','jogRightCircle4StepsRstart','jogRightCircle6StepsLstart','jogRightCircle6StepsRstart','jumpDown','jumpingJack1Reps','jumpingJack3Reps','kickLFront1Reps','kickLFront2Reps','kickLSide1Reps','kickLSide2Reps','kickRFront1Reps','kickRFront2Reps','kickRSide1Reps','kickRSide2Reps','lieDownFloor','punchLFront1Reps','punchLFront2Reps','punchLSide1Reps','punchLSide2Reps','punchRFront1Reps','punchRFront2Reps','punchRSide1Reps','punchRSide2Reps','rotateArmsBothBackward1Reps','rotateArmsBothBackward3Reps','rotateArmsBothForward1Reps','rotateArmsBothForward3Reps','rotateArmsLBackward1Reps','rotateArmsLBackward3Reps','rotateArmsLForward1Reps','rotateArmsLForward3Reps','rotateArmsRBackward1Reps','rotateArmsRBackward3Reps','rotateArmsRForward1Reps','rotateArmsRForward3Reps','runOnPlaceStartAir2StepsLStart','runOnPlaceStartAir2StepsRStart','runOnPlaceStartAir4StepsLStart','runOnPlaceStartFloor2StepsRStart','runOnPlaceStartFloor4StepsRStart','shuffle2StepsLStart','shuffle2StepsRStart','shuffle4StepsLStart','shuffle4StepsRStart','sitDownChair','sitDownFloor','sitDownKneelTieShoes','sitDownTable','skier1RepsLstart','skier3RepsLstart','sneak2StepsLStart','sneak2StepsRStart','sneak4StepsLStart','sneak4StepsRStart','squat1Reps','squat3Reps','staircaseDown3Rstart','staircaseUp3Rstart','standUpKneelToStand','standUpLieFloor','standUpSitChair','standUpSitFloor','standUpSitTable','throwBasketball','throwFarR','throwSittingHighR','throwSittingLowR','throwStandingHighR','throwStandingLowR','turnLeft','turnRight','walk2StepsLstart','walk2StepsRstart','walk4StepsLstart','walk4StepsRstart','walkBackwards2StepsRstart','walkBackwards4StepsRstart','walkLeft2Steps','walkLeft3Steps','walkLeftCircle4StepsLstart','walkLeftCircle4StepsRstart','walkLeftCircle6StepsLstart','walkLeftCircle6StepsRstart','walkOnPlace2StepsLStart','walkOnPlace2StepsRStart','walkOnPlace4StepsLStart','walkOnPlace4StepsRStart','walkRightCircle4StepsLstart','walkRightCircle4StepsRstart','walkRightCircle6StepsLstart','walkRightCircle6StepsRstart','walkRightCrossFront2Steps','walkRightCrossFront3Steps']
    actor_name = ['bd', 'bk', 'dg', 'mm', 'tr']
    max_time_num = 50
    max_frames = 0
    nums_skeleton = 31
    Samples = []
    flag_normalize = True

    Label_fn = u'./data/HDM05_dataset_sample/Labels.mat'
    data = sio.loadmat(Label_fn)
    data = data.get('Labels')
    Labels = list(data[0])
    Labels_nums = len(Labels)
    for i_Labels in range(Labels_nums):
        Labels[i_Labels] -= 1

    for i_action in action_name:
        for i_actor in actor_name:
            for i_time in range(1,max_time_num+1):
                filename = filepath + "/" +'%s_%s_%03i.mat'%(i_action, i_actor, i_time)
                flag = os.path.exists(filename)
                if flag == True:
                    data = sio.loadmat(filename)
                    frameMat = data.get('sample_mat')
                    n_frames, n_features = frameMat.shape
                    print filename
                    temp = frameMat

                    # Savitzky-Golay smoothing filter
                    if (n_frames >= 20):
                        frameMat_temp = np.zeros((n_frames - 4, n_features))
                        for i in range(2, n_frames - 2):
                            frameMat_temp[i - 2, :] = (-3 * temp[i - 2, :] + 12 * temp[i - 1, :] + 17 * temp[i,
                                                                                                    :] + 12 * temp[
                                                                                                              i + 1,
                                                                                                              :] - 3 * temp[
                                                                                                                       i + 2,
                                                                                                                       :]) / 35

                        frameMat = frameMat_temp
                    n_frames, n_features = frameMat.shape

                    # Update max_frame
                    if n_frames > max_frames:
                        max_frames = n_frames

                    # Centralization
                    temp = frameMat
                    frameMat_temp = np.zeros((n_frames, n_features))
                    for i in range(n_frames):
                        Origin = (temp[i, 0:3] + temp[i, 3:6] + temp[i, 18:21]) / 3
                        for j in range(nums_skeleton):
                            index = 3 * j
                            frameMat_temp[i, index] = temp[i, index] - Origin[0]
                            frameMat_temp[i, index + 1] = temp[i, index + 1] - Origin[1]
                            frameMat_temp[i, index + 2] = temp[i, index + 2] - Origin[2]

                    # Normalization
                    frameMat = frameMat_temp
                    if flag_normalize:
                        for j in range(n_features):
                            frameMat[:, j] = frameMat[:, j] - mean(frameMat[:, j])
                            # frameMat[:, j] = frameMat[:, j] /std(frameMat[:, j])
                            # frameMat[:, j] = np.tanh(20*frameMat[:, j])
                    Samples.append(frameMat)


    Samples = np.array(Samples)
    flag_splite5parts = True
    if flag_splite5parts:

        Samples_left_hand = []
        Samples_right_hand = []
        Samples_left_leg = []
        Samples_right_leg = []
        Samples_central_trunk = []

        nums_samples = len(Samples)
        print nums_samples
        print len(Labels)
        for i in range(nums_samples):
            sample = Samples[i]
            nums_frames = sample.shape[0]

            Frames_left_hand = []
            Frames_right_hand = []
            Frames_left_leg = []
            Frames_right_leg = []
            Frames_central_trunk = []

            for j in range(nums_frames):

                left_hand_joint = []
                for k in [18, 19, 20, 21, 22, 23, 24]:
                    left_hand_joint.extend(list(Samples[i][j][(k - 1) * 3:k * 3]))

                right_hand_joint = []
                for k in [25, 26, 27, 28, 29, 30, 31]:
                    right_hand_joint.extend(list(Samples[i][j][(k - 1) * 3:k * 3]))

                left_leg_joint = []
                for k in [13, 14, 15, 16]:
                    left_leg_joint.extend(list(Samples[i][j][(k - 1) * 3:k * 3]))

                right_leg_joint = []
                for k in [17, 18, 19, 20]:
                    right_leg_joint.extend(list(Samples[i][j][(k - 1) * 3:k * 3]))

                central_trunk_joint = []
                for k in [4, 3, 2, 1]:
                    central_trunk_joint.extend(list(Samples[i][j][(k - 1) * 3:k * 3]))

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

    print max_frames
    return results


if __name__ == '__main__':
    filepath = './data/HDM05_dataset_sample'
    [left_hand, right_hand, left_leg, right_leg, central_trunk, Labels, max_frames] = load_HDM05(filepath)
    flag_created = True
    if flag_created:
        cPickle.dump([left_hand, right_hand, left_leg, right_leg, central_trunk, Labels, max_frames],
                     open("./data/DataBackUp/HDM05_DataSet.p", "wb"))