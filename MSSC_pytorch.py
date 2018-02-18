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
filepath_train = '../dataset_padded/MSRAction3D_real_world_P4_Split_AS3_train.p'
filepath_test = '../dataset_padded/MSRAction3D_real_world_P4_Split_AS3_test.p'
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

_, time_length, n_in = skeletons_train[0].shape # time_length = number of time frames, = 67
n_res = n_in * 3 # hyperparameter
IS = 0.1
SR = 0.9 # 0.99 according to the paper, 0.9 gives best
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

# ============================== Batch processing ============================================
def get_next_batch(X, y):
	batch_X = []
	batch_y = []
	for i in range(y.shape[0]):
		batch_X.append(X[i])
		batch_y.append(y[i])
		if (i+1)%batch_size==0:
			yield(np.array(batch_X), np.array(batch_y))
			batch_X = []
			batch_y = []

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
			nn.ReLU()))

			self.LL.append(nn.Sequential( # left leg features
			nn.Conv2d(1, n_filters, (width, sliding_height), stride=strides),
			nn.ReLU()))

			self.RL.append(nn.Sequential( # right leg features
			nn.Conv2d(1, n_filters, (width, sliding_height), stride=strides),
			nn.ReLU()))

			self.Trunk.append(nn.Sequential( # central trunk features
			nn.Conv2d(1, n_filters, (width, sliding_height), stride=strides),
			nn.ReLU()))

		self.merge_hands = nn.Linear(2*n_filters*len(sliding_width), n_filters*len(sliding_width))
		self.merge_legs = nn.Linear(2*n_filters*len(sliding_width), n_filters*len(sliding_width))
		self.merge_body = nn.Linear(n_filters*len(sliding_width)*3, n_filters*len(sliding_width))
		self.final_fc = nn.Linear(n_filters*len(sliding_width), num_classes)

	def forward(self, X):
		la, ra, ll, rl, trunk = [],[],[],[],[]
		for i in range(len(sliding_width)):
			this_la = self.LA[i](X[0,:,:,:])
			this_la = F.max_pool2d(this_la, kernel_size=this_la.size()[2:]) # max pool across all time steps
			# print this_la.shape # shape=(num_examples, n_filters, 1, 1)
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
			this_trunk = self.Trunk[i](X[4,:,:,:])
			this_trunk = F.max_pool2d(this_trunk, kernel_size=this_trunk.size()[2:])
			trunk.append(this_trunk)

		la = torch.cat(la, dim=1) # shape = (num_examples, n_filters*len(sliding_width))
		ra = torch.cat(ra, dim=1) # same shape as above
		hand_features = self.merge_hands(torch.cat([la, ra], dim=1)[:,:,0,0]) # shape = (num_examples, n_filters*len(sliding_width))
		
		ll = torch.cat(ll, dim=1)
		rl = torch.cat(rl, dim=1)
		leg_features = self.merge_legs(torch.cat([ll, rl], dim=1)[:,:,0,0])

		trunk_features = torch.cat(trunk, dim=1)[:,:,0,0] # shape = (num_examples, n_filters*len(sliding_width))

		body_features = self.merge_body(torch.cat([hand_features, leg_features, trunk_features], dim=1)) # shape = (num_examples, n_filters*len(sliding_width))

		output = self.final_fc(body_features)
		return output

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Linear') != -1: # lecun uniform initialization
		m.weight.data.uniform_(-np.sqrt(3. / m.in_features), np.sqrt(3. / m.in_features))

# ================================ Training =========================================
model = Decoder()
model.apply(weights_init)

criterion = nn.CrossEntropyLoss() # loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Adam optimizer as in paper
echo_states_train = np.transpose(echo_states_train, (1, 0, 2, 3, 4))
for epoch in range(n_epochs):
	shuffle(echo_states_train, labels_train)
	for iteratn, (batch_X, batch_y) in enumerate(get_next_batch(echo_states_train, labels_train)):
		batch_X = np.transpose(batch_X, (1, 0, 2, 3, 4))
		X = Variable(torch.from_numpy(batch_X))
		y = Variable(torch.from_numpy(batch_y).long())
		optimizer.zero_grad() # clears the gradients of all optimized Variable
		predictions = model(X)
		loss = criterion(predictions, y) # y should NOT be one-hot
		loss.backward()
		optimizer.step()
		print 'Epoch', epoch, 'Iteration', iteratn, 'Loss', loss.data[0] # Final loss is around 0.159

# ============================= Training evaluation ==================================
echo_states_train = np.transpose(echo_states_train, (1, 0, 2, 3, 4))
X = Variable(torch.from_numpy(echo_states_train))
y_train = torch.from_numpy(labels_train).long()
predictions = model(X)
_, predictions = torch.max(predictions.data, 1)
accuracy = (predictions == y_train).sum()
print 'Train Accuracy', accuracy*1.0/num_samples_train # Train accuracy is approx 99.24%

# ================================ Testing ===========================================
X_test = Variable(torch.from_numpy(np.array(echo_states_test)))
y_test = torch.from_numpy(labels_test).long()
model.eval()
predictions = model(X_test)
_, predictions = torch.max(predictions.data, 1)
accuracy = (predictions == y_test).sum()
print 'Test Accuracy', accuracy*1.0/num_samples_test # Test accuracy is approx 83.9%
