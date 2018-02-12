import numpy as np
import cPickle as cp

import torch 
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F

from sklearn.utils import shuffle

import reservoir
import utils

print 'Loading data...'
"""
a .p file is a list: [left_hand_skeleton, right_hand_skeleton, left_leg_skeleton, right_leg_skeleton, central_trunk_skeleton, labels]
the shape of the first five ones: (num_samples, time_length, num_joints)
the shape of the last one: (num_samples,)
"""
filepath_train = '../dataset/MSRAction3D_real_world_P4_Split_AS3_train.p'
filepath_test = '../dataset/MSRAction3D_real_world_P4_Split_AS3_test.p'
data_train = cp.load(open(filepath_train, 'rb'))
skeletons_train = data_train[0:5]
labels_train = data_train[5]
data_test = cp.load(open(filepath_test, 'rb'))
skeletons_test = data_test[0:5]
labels_test = data_test[5]

print 'Transfering labels...'
labels_train, labels_test, num_classes = utils.transfer_labels(labels_train, labels_test)

# ============================= Reservoir initialization =================================
num_samples_train = labels_train.shape[0]
num_samples_test = labels_test.shape[0]

_, time_length, n_in = skeletons_train[0].shape
n_res = n_in * 3
IS = 0.1
SR = 0.9
sparsity = 0.3
leakyrate = 1.0

#create five different reservoirs, one for a skeleton part
reservoirs = [reservoir.reservoir_layer(n_in, n_res, IS, SR, sparsity, leakyrate) for i in range(5)]

# Get all the ESRs for each of the input (same as input shape, plus one added dimension of only 1 column)
print 'Getting echo states...'
echo_states_train = [np.empty((num_samples_train, 1, time_length, n_res), np.float32) for i in range(5)]
echo_states_test = [np.empty((num_samples_test, 1, time_length, n_res), np.float32) for i in range(5)]
for i in range(5):
	echo_states_train[i][:,0,:,:] = reservoirs[i].get_echo_states(skeletons_train[i])
	echo_states_test[i][:,0,:,:] = reservoirs[i].get_echo_states(skeletons_test[i])
echo_states_train = [np.concatenate(echo_states_train[0:2], axis = 1), np.concatenate(echo_states_train[2:4], axis = 1), echo_states_train[4]]
echo_states_test = [np.concatenate(echo_states_test[0:2], axis = 1), np.concatenate(echo_states_test[2:4], axis = 1), echo_states_test[4]]

# ======================= hyperparameters for the conv-decoder part =====================================
input_shapes = ((2, time_length, n_res), (2, time_length, n_res), (1, time_length, n_res))
sliding_width = [2, 3, 4] # width of sliding conv windows -> extracts multi-scale features
sliding_height = n_res # height of sliding conv windows
n_filters = 16 # number of filters under each width
strides = (1, 1)
batch_size = 8
n_epochs = 300
learning_rate = 0.001

# ============================== Batch processing ============================================
def get_next_batch(X, y):
	batch_arms = []
	batch_legs = []
	batch_trunk = []
	batch_y = []
	for i in range(y.shape[0]):
		batch_arms.append(X[0][i])
		batch_legs.append(X[1][i])
		batch_trunk.append(X[2][i])
		batch_y.append(y[i])
		if (i+1)%batch_size==0:
			yield(np.array(batch_arms), np.array(batch_legs), np.array(batch_trunk), np.array(batch_y))
			batch_arms = []
			batch_legs = []
			batch_trunk = []
			batch_y = []

# ============================== Conv-decoder ================================================
class Decoder(nn.Module):
	def __init__(self):
		super(Decoder, self).__init__()
		self.Arms, self.Legs, self.Trunk = [],[],[]
		for width in sliding_width:
			self.Arms.append(nn.Sequential( # arm features
			nn.Conv2d(2, n_filters, (width, sliding_height), stride=strides),
			nn.ReLU()))

			self.Legs.append(nn.Sequential( # leg features
			nn.Conv2d(2, n_filters, (width, sliding_height), stride=strides),
			nn.ReLU()))

			self.Trunk.append(nn.Sequential( # central trunk features
			nn.Conv2d(1, n_filters, (width, sliding_height), stride=strides),
			nn.ReLU()))

		self.merge_body = nn.Linear(n_filters*len(sliding_width)*3, n_filters*len(sliding_width))
		self.final_fc = nn.Linear(n_filters*len(sliding_width), num_classes)

	def forward(self, X_arms, X_legs, X_trunk):
		arms, legs, trunk = [],[],[]
		for i in range(len(sliding_width)):
			this_arm = self.Arms[i](X_arms)
			this_arm = F.max_pool2d(this_arm, kernel_size=this_arm.size()[2:]) # max pool across all time steps
			# print this_arm.shape # shape=(num_examples, n_filters, 1, 1)
			arms.append(this_arm)

			this_leg = self.Legs[i](X_legs)
			this_leg = F.max_pool2d(this_leg, kernel_size=this_leg.size()[2:])
			legs.append(this_leg)

			this_trunk = self.Trunk[i](X_trunk)
			this_trunk = F.max_pool2d(this_trunk, kernel_size=this_trunk.size()[2:])
			trunk.append(this_trunk)

		hand_features = torch.cat(arms, dim=1)
		leg_features = torch.cat(legs, dim=1)
		trunk_features = torch.cat(trunk, dim=1)
		body_features = torch.cat([hand_features, leg_features, trunk_features], dim=1)[:,:,0,0]

		body_features = self.merge_body(body_features)
		outputs = self.final_fc(body_features)
		return outputs

# ================================ Training =========================================
model = Decoder()
criterion = nn.CrossEntropyLoss() # loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Adam optimizer as in paper
for epoch in range(n_epochs):
	for iteratn, (batch_arms, batch_legs, batch_trunk, batch_y) in enumerate(get_next_batch(echo_states_train, labels_train)):
		X_arms = Variable(torch.from_numpy(batch_arms)) # shape=(n_examples, 2, time_frames, n_res)
		X_legs = Variable(torch.from_numpy(batch_legs)) # shape=(n_examples, 2, time_frames, n_res)
		X_trunk = Variable(torch.from_numpy(batch_trunk)) # shape=(n_examples, 1, time_frames, n_res)
		y = Variable(torch.from_numpy(batch_y).long())
		optimizer.zero_grad() # clears the gradients of all optimized Variable
		predictions = model(X_arms, X_legs, X_trunk)
		loss = criterion(predictions, y) # y should NOT be one-hot
		loss.backward()
		optimizer.step()
		print 'Epoch', epoch, 'Iteration', iteratn, 'Loss', loss.data[0] # Final loss is around 0.5823

# ============================= Training evaluation ==================================
X_arms = Variable(torch.from_numpy(np.array(echo_states_train[0])))
X_legs = Variable(torch.from_numpy(np.array(echo_states_train[1])))
X_trunk = Variable(torch.from_numpy(np.array(echo_states_train[2])))
y_train = torch.from_numpy(labels_train).long()
predictions = model(X_arms, X_legs, X_trunk)
_, predictions = torch.max(predictions.data, 1)
accuracy = (predictions == y_train).sum()
print 'Train Accuracy', accuracy*1.0/num_samples_train # Train accuracy is approx 95.61%

# ================================ Testing ===========================================
X_arms = Variable(torch.from_numpy(np.array(echo_states_test[0]))) # shape=(n_examples, 2, time_frames, n_res)
X_legs = Variable(torch.from_numpy(np.array(echo_states_test[1]))) # shape=(n_examples, 2, time_frames, n_res)
X_trunk = Variable(torch.from_numpy(np.array(echo_states_test[2]))) # shape=(n_examples, 1, time_frames, n_res)
y_test = torch.from_numpy(labels_test).long()
model.eval()
predictions = model(X_arms, X_legs, X_trunk)
_, predictions = torch.max(predictions.data, 1)
accuracy = (predictions == y_test).sum()
print 'Test Accuracy', accuracy*1.0/num_samples_test # Test accuracy is approx 92.85%
