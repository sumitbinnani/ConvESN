import numpy as np
from keras.utils import np_utils

#make the labels be in [0, num_classes-1] and transfer them to one-hot vectors
def transfer_labels(labels_train, labels_test):

	indexes = np.unique(labels_train)

	num_classes = indexes.shape[0]
	num_samples_train = labels_train.shape[0]
	num_samples_test = labels_test.shape[0]

	for i in range(num_samples_train):
		new_label = np.argwhere(indexes == labels_train[i])[0][0]
		labels_train[i] = new_label
	labels_train = np_utils.to_categorical(labels_train, num_classes)

	for i in range(num_samples_test):
		new_label = np.argwhere(indexes == labels_test[i])[0][0]
		labels_test[i] = new_label
	labels_test = np_utils.to_categorical(labels_test, num_classes)
	
	return labels_train, labels_test, num_classes
