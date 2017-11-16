import numpy as np

import scipy as sp
from scipy.sparse import *
from scipy.sparse.linalg import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class reservoir_layer(object):
	
	def __init__(self, n_in, n_res, IS, SR, sparsity, leakyrate, use_bias = False):
		
		self.n_in = n_in #the number of input units
		self.n_res = n_res #the number of reservoir units
		self.IS = IS #the input scale
		self.SR = SR #the spectral radius of W_res matrix
		self.sparsity = sparsity #the sparsity of W_res matrix
		self.leakyrate = leakyrate #the leakyrate used when update the echo state
		self.use_bias = use_bias #whether to use bias

		self.W_in = 2 * np.random.random(size = (self.n_res, self.n_in)) - 1 #input-to-state weight matrix

		W_res_temp = sp.sparse.rand(self.n_res, self.n_res, self.sparsity)
		vals, vecs = sp.sparse.linalg.eigsh(W_res_temp, k = 1)
		self.W_res = (self.SR * W_res_temp / vals[0]).toarray() #recurrent weight matrix

		b_bound = 0.1
		self.b = 2 * b_bound * np.random.random(size = (self.n_res)) - b_bound
	
	#get all the echo states except the first n_forget_steps ones
	def get_echo_states(self, series, n_forget_steps = 0):
		
		num_samples, time_length, _ = series.shape

		echo_states = np.empty((num_samples, time_length - n_forget_steps, self.n_res), np.float32)

		for i in range(num_samples):
			
			collect_states = np.empty((time_length - n_forget_steps, self.n_res), np.float32)
			x = np.zeros((self.n_res))

			for t in range(time_length):
				
				u = series[i, t]

				if self.use_bias:
					xUpd = np.tanh(np.dot(self.W_in, self.IS * u) + np.dot(self.W_res, x) + self.b)
				else:
					xUpd = np.tanh(np.dot(self.W_in, self.IS * u) + np.dot(self.W_res, x))

				x = (1 - self.leakyrate) * x + self.leakyrate * xUpd

				if t >= n_forget_steps:
					collect_states[t - n_forget_steps] = x

			echo_states[i] = collect_states

		return echo_states

#visualize the echo state and save the picture
def visualize_echo_state(echo_state, filename):

	plt.figure(figsize = (19.20, 10.80))

	time_length, n_res = echo_state.shape

	x_labels = range(n_res)
	y_labels = range(time_length)

	x_ticks = np.array(range(len(x_labels)))
	y_ticks = np.array(range(len(y_labels)))

	plt.gca().set_xticks(x_ticks, minor = True)
	plt.gca().set_yticks(y_ticks, minor = True)
	plt.gca().xaxis.set_ticks_position('none')
	plt.gca().yaxis.set_ticks_position('none')

	plt.grid(None, which = 'minor', linestyle = 'none')

	plt.imshow(echo_state.T, interpolation = 'nearest', aspect = 'auto')

	plt.colorbar()

	plt.xlabel('Time Direction')
	plt.ylabel('Reservoir Neurons')

	plt.savefig(filename)

	plt.close('all')
