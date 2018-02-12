import numpy as np
import cPickle as cp

import torch 
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F

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

_, time_length, n_in = skeletons_train[0].shape # time_length = number of time frames, =67
n_res = n_in * 3 # hyperparameter
IS = 0.1
SR = 0.99 # 0.99 according to the paper
sparsity = 0.3 # isnt this alpha in the paper, then it should be 0.01
leakyrate = 1.0

# Create five different reservoirs, one for a skeleton part
reservoirs = [reservoir.reservoir_layer(n_in, n_res, IS, SR, sparsity, leakyrate) for i in range(5)]

# Get all the ESRs for each of the input (same as input shape, plus one added dimension of only 1 column)
print 'Getting echo states...'
echo_states_train = [np.empty((num_samples_train, 1, time_length, n_res), np.float32) for i in range(5)]
echo_states_test = [np.empty((num_samples_test, 1, time_length, n_res), np.float32) for i in range(5)]
for i in range(5):
	echo_states_train[i][:,0,:,:] = reservoirs[i].get_echo_states(skeletons_train[i])
	echo_states_test[i][:,0,:,:] = reservoirs[i].get_echo_states(skeletons_test[i])


# ======================= hyperparameters for the conv-decoder part =====================================

input_shape = (1, time_length, n_res)
sliding_width = [2, 3, 4] # width of sliding conv windows -> extracts multi-scale features
sliding_height = n_res # height of sliding conv windows
n_filters = 16 # number of filters under each width
strides = (1, 1)
batch_size = 8
n_epochs = 300
learning_rate = 0.001

# ============================== Conv-decoder ================================================
class Decoder(nn.Module):
	def __init__(self):
		super(Decoder, self).__init__()
		self.LA, self.RA, self.LL, self.RL, self.Trunk = [],[],[],[],[]
		for width in sliding_width:
			self.LA.append(nn.Sequential( # left arm features
			nn.Conv2d(1, n_filters, (width, sliding_height), stride=strides),
			nn.ReLU()))

			self.RA.append(nn.Sequential( # right arm features
			nn.Conv2d(1, n_filters, (width, sliding_height), stride=strides),
			nn.ReLU(),
			nn.MaxPool2d(2)))

			self.LL.append(nn.Sequential( # left leg features
			nn.Conv2d(1, n_filters, (sliding_width[0], sliding_height), stride=strides),
			nn.ReLU(),
			nn.MaxPool2d(2)))

			self.RL.append(nn.Sequential( # right leg features
			nn.Conv2d(1, n_filters, (sliding_width[0], sliding_height), stride=strides),
			nn.ReLU(),
			nn.MaxPool2d(2)))

			self.Trunk.append(nn.Sequential( # central trunk features
			nn.Conv2d(1, n_filters, (sliding_width[0], sliding_height), stride=strides),
			nn.ReLU(),
			nn.MaxPool2d(2)))


	def forward(self, X):
		la, ra, ll, rl, trunk = [],[],[],[],[]
		for i in range(len(sliding_width)):
			this_la = self.LA[i](X[0,:,:,:])
			this_la = F.max_pool2d(this_la, kernel_size=this_la.size()[2:]) # max pool across all time steps
			la.append(this_la)
			this_ra = self.RA[i](X[1,:,:,:])
			this_ra = F.max_pool2d(this_ra, kernel_size=this_ra.size()[2:])
			ra.append(this_ra)
			this_ll = self.LL[i](X[2,:,:,:])
			this_ll = F.max_pool2d(this_ll, kernel_size=this_ll.size()[2:])
			ll.append(this_ll)
			this_rl = self.RL[i](X[3,:,:,:])
			this_rl = F.max_pool2d(this_rl, kernel_size=this_rl.size()[2:])
			rl.append(this_rl)

		# hand_features = 

# =========================== Training and Testing =========================================
model = Decoder()
criterion = nn.CrossEntropyLoss() # loss
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Adam optimizer as in paper
X = Variable(torch.from_numpy(np.array(echo_states_train))) # shape=(5, n_examples, 1, time_frames, n_res)
y = Variable(torch.from_numpy(labels_train))
X_test = Variable(torch.from_numpy(np.array(echo_states_test)))
y_test = Variable(torch.from_numpy(labels_test))
for epoch in range(n_epochs):
	optimizer.zero_grad() # clears the gradients of all optimized Variable
	outputs = model(X)
