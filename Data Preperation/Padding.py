import cPickle
import numpy as np
import matplotlib.pyplot as plt


def process(data_train, data_test):
    train_left_hand, train_right_hand, train_left_leg, train_right_leg, train_central_trunk, train_Labels, train_max_frames = data_train
    test_left_hand, test_right_hand, test_left_leg, test_right_leg, test_central_trunk, test_Labels, test_max_frames = data_test

    max_frames = max(train_max_frames, test_max_frames)

    print "Preprocessing training data ..."
    train_left_hand  	= padding(train_left_hand , max_frame = max_frames)
    train_right_hand 	= padding(train_right_hand, max_frame = max_frames)
    train_left_leg   	= padding(train_left_leg  , max_frame = max_frames)
    train_right_leg  	= padding(train_right_leg , max_frame = max_frames)
    train_central_trunk = padding(train_central_trunk, max_frame = max_frames)

    print "Preprocessing testing data ..."
    test_left_hand  	= padding(test_left_hand 	, max_frame = max_frames)
    test_right_hand 	= padding(test_right_hand   , max_frame = max_frames)
    test_left_leg   	= padding(test_left_leg  	, max_frame = max_frames)
    test_right_leg  	= padding(test_right_leg 	, max_frame = max_frames)
    test_central_trunk 	= padding(test_central_trunk, max_frame = max_frames)

    print "Preprocessing finished ..."

    data_train = [train_left_hand, train_right_hand, train_left_leg, train_right_leg, train_central_trunk, train_Labels]
    data_test = [test_left_hand, test_right_hand, test_left_leg, test_right_leg, test_central_trunk, test_Labels]

    return data_train, data_test


def process_singal(data):
    left_hand, right_hand, left_leg, right_leg, central_trunk, Labels, train_max_frames = data

    max_frames = train_max_frames

    print "Preprocessing training data ..."
    left_hand  	= padding(left_hand , max_frame = max_frames)
    right_hand 	= padding(right_hand, max_frame = max_frames)
    left_leg   	= padding(left_leg  , max_frame = max_frames)
    right_leg  	= padding(right_leg , max_frame = max_frames)
    central_trunk = padding(central_trunk, max_frame = max_frames)

    print "Preprocessing finished ..."

    data = [left_hand, right_hand, left_leg, right_leg, central_trunk, Labels]
    return data


def padding(data, max_frame=300):

    nums_sample = data.shape[0]
    nums_data_frame, nums_feature = np.shape(data[0])

    data_new = []

    for i in range(nums_sample):
        sample_i = np.array(data[i])

        num_frame_sample_i = np.shape(sample_i)[0]

        assert max_frame >= num_frame_sample_i

        zero = np.zeros((max_frame - num_frame_sample_i, nums_feature))
        sample_i_new = np.vstack((sample_i, zero))

        data_new.append(sample_i_new)

    data = np.array(data_new)
    return data


if __name__ == '__main__':
    protocol = "P4"
    print "-------------------------------------------------------------------------------"
    print "Loading AS1_train data ...."
    P4_AS1_train = cPickle.load(
        open("./data/DataBackUp/MSRAction3D_real_world_" + protocol + "_Split_AS1_train.p", "rb"))
    print "Loading AS1_test  data ...."
    P4_AS1_test = cPickle.load(
        open("./data/DataBackUp/MSRAction3D_real_world_" + protocol + "_Split_AS1_test.p", "rb"))
    print "Loading AS2_train  data ...."
    P4_AS2_train = cPickle.load(
        open("./data/DataBackUp/MSRAction3D_real_world_" + protocol + "_Split_AS2_train.p", "rb"))
    print "Loading AS2_test  data ...."
    P4_AS2_test = cPickle.load(
        open("./data/DataBackUp/MSRAction3D_real_world_" + protocol + "_Split_AS2_test.p", "rb"))
    print "Loading AS3_train  data ...."
    P4_AS3_train = cPickle.load(
        open("./data/DataBackUp/MSRAction3D_real_world_" + protocol + "_Split_AS3_train.p", "rb"))
    print "Loading AS3_test  data ...."
    P4_AS3_test = cPickle.load(
        open("./data/DataBackUp/MSRAction3D_real_world_" + protocol + "_Split_AS3_test.p", "rb"))
    AS1_train, AS1_test = process(P4_AS1_train, P4_AS1_test)
    AS2_train, AS2_test = process(P4_AS2_train, P4_AS2_test)
    AS3_train, AS3_test = process(P4_AS3_train, P4_AS3_test)
    AS1_train_left_hand, AS1_train_right_hand, AS1_train_left_leg, AS1_train_right_leg, AS1_train_central_trunk, AS1_train_Labels = AS1_train
    AS1_test_left_hand, AS1_test_right_hand, AS1_test_left_leg, AS1_test_right_leg, AS1_test_central_trunk, AS1_test_Labels = AS1_test
    AS2_train_left_hand, AS2_train_right_hand, AS2_train_left_leg, AS2_train_right_leg, AS2_train_central_trunk, AS2_train_Labels = AS2_train
    AS2_test_left_hand, AS2_test_right_hand, AS2_test_left_leg, AS2_test_right_leg, AS2_test_central_trunk, AS2_test_Labels = AS2_test
    AS3_train_left_hand, AS3_train_right_hand, AS3_train_left_leg, AS3_train_right_leg, AS3_train_central_trunk, AS3_train_Labels = AS3_train
    AS3_test_left_hand, AS3_test_right_hand, AS3_test_left_leg, AS3_test_right_leg, AS3_test_central_trunk, AS3_test_Labels = AS3_test
    cPickle.dump(
        [AS1_train_left_hand, AS1_train_right_hand, AS1_train_left_leg, AS1_train_right_leg, AS1_train_central_trunk,
         AS1_train_Labels], open("./data/MSRAction3D_real_world_" + protocol + "_Split_AS1_train.p", "wb"))
    cPickle.dump(
        [AS2_train_left_hand, AS2_train_right_hand, AS2_train_left_leg, AS2_train_right_leg, AS2_train_central_trunk,
         AS2_train_Labels], open("./data/MSRAction3D_real_world_" + protocol + "_Split_AS2_train.p", "wb"))
    cPickle.dump(
        [AS3_train_left_hand, AS3_train_right_hand, AS3_train_left_leg, AS3_train_right_leg, AS3_train_central_trunk,
         AS3_train_Labels], open("./data/MSRAction3D_real_world_" + protocol + "_Split_AS3_train.p", "wb"))
    cPickle.dump(
        [AS1_test_left_hand, AS1_test_right_hand, AS1_test_left_leg, AS1_test_right_leg, AS1_test_central_trunk,
         AS1_test_Labels], open("./data/MSRAction3D_real_world_" + protocol + "_Split_AS1_test.p", "wb"))
    cPickle.dump(
        [AS2_test_left_hand, AS2_test_right_hand, AS2_test_left_leg, AS2_test_right_leg, AS2_test_central_trunk,
         AS2_test_Labels], open("./data/MSRAction3D_real_world_" + protocol + "_Split_AS2_test.p", "wb"))
    cPickle.dump(
        [AS3_test_left_hand, AS3_test_right_hand, AS3_test_left_leg, AS3_test_right_leg, AS3_test_central_trunk,
         AS3_test_Labels], open("./data/MSRAction3D_real_world_" + protocol + "_Split_AS3_test.p", "wb"))
    print "MSRAction3D_real_world_P4 loading finished!"


    print "-------------------------------------------------------------------------------"
    print "Loading Florence3D_DataSet ...."
    Florence3D_all = []
    for i in range(10):
        Florence3D_DataSet = cPickle.load(open("./data/DataBackUp/Florence3D_DataSet.p", "rb"))
        Florence3D_all_temp = process_singal(Florence3D_DataSet[i])
        left_hand, right_hand, left_leg, right_leg, central_trunk, Labels = Florence3D_all_temp
        Florence3D_all.append([left_hand, right_hand, left_leg, right_leg, central_trunk, Labels])
    cPickle.dump(Florence3D_all,open("./data/Florence3D_DataSet.p", "wb"))
    print "Florence3D_DataSet created!"
    print "-------------------------------------------------------------------------------"


    print "-------------------------------------------------------------------------------"
    print "Loading UTKinect_DataSet ...."
    UTKinect_DataSet = cPickle.load(open("./data/DataBackUp/UTKinect_DataSet.p", "rb"))
    UTKinect_all = process_singal(UTKinect_DataSet)
    left_hand, right_hand, left_leg, right_leg, central_trunk, Labels = UTKinect_all
    cPickle.dump(
        [left_hand, right_hand, left_leg, right_leg, central_trunk, Labels], open("./data/UTKinect_DataSet.p", "wb"))
    print "UTKinect_DataSet created!"
    print "-------------------------------------------------------------------------------"



    print "-------------------------------------------------------------------------------"
    print "Loading HDM05_DataSet ...."
    HDM05_DataSet = cPickle.load(open("./data/DataBackUp/HDM05_DataSet.p", "rb"))
    HDM05_all = process_singal(HDM05_DataSet)
    left_hand, right_hand, left_leg, right_leg, central_trunk, Labels = HDM05_all
    cPickle.dump(
        [left_hand, right_hand, left_leg, right_leg, central_trunk, Labels], open("./data/HDM05_DataSet.p", "wb"))
    print "HDM05_DataSet created!"
    print "-------------------------------------------------------------------------------"






