import os
import csv
from numpy import *
import numpy as np
import cPickle
import random


def load_UTKinect_PreProcess (filepath,params):

    max_frames = 0
    nums_skeleton = 20
    flag_normalize = True
    Labels = []
    Samples = []
    subjects = params[20]
    times = params[21]
    count = -1

    for s in subjects:
        for t in times:
            fileName = filepath + "/" + 'joints_s%02i_e%02i.txt' % (s, t)
            frame_no = []
            sample_mat = []
            flag = os.path.exists(fileName)
            if flag == True:
                count = count+1
                fr = open(fileName)
                for line in fr.readlines():
                    lineArr = line.strip().split('   ')
                    frame_no.extend([float(lineArr[0])])
                    frame_mat = []
                    for i in range(60):
                        frame_mat.extend([float(lineArr[1].strip().split('  ')[i])])
                    sample_mat.append(frame_mat)
                frame_no = np.array(frame_no)
                frame, frame_no_index = np.unique(frame_no, return_index=True)
                sample_mat = np.array(sample_mat)

                for i in range(10):
                    if (math.isnan(params[count][i][0])) or (math.isnan(params[count][i][1])):
                        continue
                    else:
                        ind1 = np.argmin(abs(frame - params[count][i][0]))
                        ind2 = np.argmin(abs(frame - params[count][i][1]))

                        action = []
                        for j in range(frame_no_index[ind1], frame_no_index[ind2 + 1]):
                            action_one_frame = list(sample_mat[j][:])
                            action.append(action_one_frame)
                        frameMat = action
                        frameMat = np.array(frameMat)

                        # Truncation
                        n_frames, n_features = frameMat.shape
                        if n_frames>100:
                            continue

                        temp = frameMat
                        # Savitzky-Golay smoothing filter
                        if(n_frames>=30):
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

                        # Update max_frame
                        if n_frames > max_frames:
                            max_frames = n_frames

                        # Centralization
                        temp = frameMat
                        frameMat_temp = np.zeros((n_frames, n_features))

                        for p in range(n_frames):
                            Origin = (temp[p, 0:3] + temp[p, 36:39] + temp[p, 48:51]) / 3
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
                                # frameMat[:, j] = frameMat[:, j] /std(frameMat[:, j])
                                # frameMat[:, j] = np.tanh(20 * frameMat[:, j])

                        Labels.extend([i])
                        Samples.append(frameMat)
    Samples_nums = len(Samples)
    randnums = range(Samples_nums)
    random.shuffle(randnums)
    Samples_temp = Samples
    Labels_temp = Labels
    j = 0
    for i in randnums:
        Samples[j] = Samples_temp[i]
        Labels[j] = Labels_temp[i]
        j += 1

    Samples = np.array(Samples)
    print Samples
    print Labels
    print len(Samples), len(Labels)
    flag_splite5parts = True
    if flag_splite5parts:

        Samples_left_hand = []
        Samples_right_hand = []
        Samples_left_leg = []
        Samples_right_leg = []
        Samples_central_trunk = []

        nums_samples = len(Samples)

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
                for k in [5, 6, 7, 8]:
                    left_hand_joint.extend(list(Samples[i][j][(k - 1) * 3:k * 3]))

                right_hand_joint = []
                for k in [9, 10, 11, 12]:
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
    return results

if __name__ == '__main__':
    filepath = "./data/UTKinect"
    joints_s01_e01_action_frame=[[252,390],[572,686],[704,752],[822,954],[1016,1242],[1434,1488],[1686,1748],[1640,1686],[1834,2064],[2110,2228]]
    joints_s01_e02_action_frame = [[154,192], [530,628], [640,720], [1202,1356], [1364,1520], [2246,2294], [2752,2792], [2820,2858], [2984,3204], [3250,3448]]
    joints_s02_e01_action_frame = [[266,368], [672,788], [818,910], [1262,1386], [1424,1780], [2040,2086], [2340,2376], [2488,2550], [2668,2830], [3198,3324]]
    joints_s02_e02_action_frame = [[40,208], [468,602], [620,722], [894,1038], [1340,1480], [1966,2014], [2194,2230], [2314,2358], [2408,2630], [2690,2810]]
    joints_s03_e01_action_frame = [[372,528], [734,862], [902,1000], [1118,1284], [1934,2168], [3226,3282], [3556,3622], [3660,3730], [3806,3960], [4076,4184]]
    joints_s03_e02_action_frame = [[122,254], [452,592], [644,724], [848,1018], [1078,1192], [1638,1690], [1866,1896], [1928,2008], [2054,2208], [2324,2460]]
    joints_s04_e01_action_frame = [[348,496], [788,864], [954,1056], [1190,1326], [1580,1882], [2306,2350], [2532,2572], [2644,2686], [2790,2968], [3064,3146]]
    joints_s04_e02_action_frame = [[420,546], [1046,1144], [1352,1414], [1682,1820], [1868,2122], [2564,2608], [2760,2792], [2866,2910], [3070,3260], [3448,3622]]
    joints_s05_e01_action_frame = [[708,888], [1140,1238], [1294,1394], [1482,1676], [1736,2064], [3104,3176], [3596,3632], [3706,3770], [3946,4352], [4522,4734]]
    joints_s05_e02_action_frame = [[212,376], [634,756], [788,862], [974,1180], [1266,1540], [1752,1828], [2172,2230], [2104,2156], [2504,2784], [2798,2900]]
    joints_s06_e01_action_frame = [[1230,1366], [1564,1644], [1678,1758], [1862,1948], [1966,2098], [2392,2414], [2672,2698], [2790,2824], [3046,3216], [3290,3444]]
    joints_s06_e02_action_frame = [[294,426], [710,818], [856,956], [1088,1174], [2031,2202], [2518,2562], [2702,2726], [2770,2808], [2952,3060], [3096,3188]]
    joints_s07_e01_action_frame = [[130,252], [1038,1186], [1256,1372], [1450,1602], [1602,1758], [2534,2614], [3290,3350], [3350,3522], [3666,3902], [3990,4128]]
    joints_s07_e02_action_frame = [[552,638], [880,1014], [1014,1146], [1228,1352], [1352,1518], [1990,2058], [2434,2496], [2496,2618], [2672,2982], [3042,3152]]
    joints_s08_e01_action_frame = [[446,534], [714,812], [836,900], [1026,1144], [1228,1588], [1880,1916], [2236,2268], [2334,2398], [2598,2772], [2794,2892]]
    joints_s08_e02_action_frame = [[138,246], [610,716], [770,878], [1126,1200], [1364,1650], [1826,1878], [2030,2078], [2126,2204], [2280,2506], [2574,2650]]
    joints_s09_e01_action_frame = [[404,544], [1080,1196], [1212,1290], [1422,1538], [1668,1970], [2688,2728], [3266,3316], [3316,3390], [3576,3762], [3992,4118]]
    joints_s09_e02_action_frame = [[482,610], [1026,1158], [1206,1310], [1546,1678], [1714,2120], [2468,2522], [2696,2760], [2770,2838], [4708,4872], [4904,4964]]
    joints_s10_e01_action_frame = [[100,272], [562,730], [730,862], [924,1150], [1394,1846], [3304,3388], [3468,3524], [3524,3608], [3962,4222], [4268,4336]]
    joints_s10_e02_action_frame = [[96,220], [500,658], [664,770], [1022,1232], [NaN,NaN], [1720,1810], [1944,1994], [1982,2062], [2094,2350], [2454,2598]]
    paramas=[joints_s01_e01_action_frame,joints_s01_e02_action_frame,joints_s02_e01_action_frame,joints_s02_e02_action_frame,joints_s03_e01_action_frame,joints_s03_e02_action_frame,joints_s04_e01_action_frame,joints_s04_e02_action_frame,joints_s05_e01_action_frame,joints_s05_e02_action_frame,joints_s06_e01_action_frame,joints_s06_e02_action_frame,joints_s07_e01_action_frame,joints_s07_e02_action_frame,joints_s08_e01_action_frame,joints_s08_e02_action_frame,joints_s09_e01_action_frame,joints_s09_e02_action_frame,joints_s10_e01_action_frame,joints_s10_e02_action_frame]
    paramas.append(range(1, 11))
    paramas.append(range(1, 3))
    [left_hand, right_hand, left_leg, right_leg, central_trunk, Labels, max_frames] = load_UTKinect_PreProcess(filepath,paramas)

    flag_created = True
    if flag_created:
        cPickle.dump([left_hand, right_hand, left_leg, right_leg, central_trunk, Labels, max_frames], open("./data/DataBackUp/UTKinect_DataSet.p", "wb"))