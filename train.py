import os
import time
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import DataSet
from model import Down1d, Up1d, Bottleneck, AudioUnet
from utility import avg_sqrt_l2_loss

import librosa

default_opt = {'alg': 'adam', 'lr': 1e-4, 'b1': 0.99, 'b2': 0.999, 'num_layers': 4, 'batch_size': 128}

class Solver(object):
	def __init__(self, data_loader, config):

		self.config = config
		self.data_loader = data_loader

		self.alg = self.config['alg']
		self.lr = self.config['lr']
		self.betas = (self.config['b1'], self.config['b2'])
		self.num_layers = self.config['num_layers']
		self.batch_size = self.config['batch_size']
		self.log_dir = self.config['log_dir']
		self.epoch = self.config['epoch']
		self.model_save_dir = self.config['model_save_dir']
		self.model_save_step = self.config['model_save_step']

	def build_model(self):
		device = torch.device("cuda:0" if torch.cuda.is_avaialble() else "cpu")

		self.model = AudioUnet(self.num_layers)

		#Data Parallel
		# if torch.cuda.device_count() > 1:
		# 	model = nn.DataParallel(model)
		# model.to(device)

		if self.alg = "adam":
			self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr, betas=self.betas)
		else
			raise ValueError('Invalid Optimizer: ' + self.alg)


	def print_network(self):
		"""Print out the network information"""
		num_params = 0
		for p in self.model.parameters():
			num_params += p.numel()
		print(self.model)
		print("The number of parameters: {}".format(num_params))

	def build_tensorboard(self):
		"""Build a tensorboard logger."""
		from logger import Logger
		self.logger = Logger(self.log_dir)

	def create_objective(self, X, Y, ):
		"""Not Implemented Most prolly inside create model routine"""

	def train(self):
		build_model()

		#train Loops
		for epoch in range(self.epoch):
			for data in self.data_loader:
				data.to(device)

				output = self.model(data)
				loss = avg_sqrt_l2_loss(data, output) 
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

			print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, loss.data[0]))

			#tensorboard steps

			#checkpoint the model
			if (epoch+1) % self.model_save_step == 0:
				model_path = os.path.join(self.model_save_dir, '{}-model.ckpt'.format(epoch+1))
				torch.save(self.model.state_dict(), model_path)
				print('Saved Model checkpoints into {}'.format(self.model_save_dir))

		torch.save(self.model.state_dict(), './AudioUnet.pth')


		



	def load_data(self):
		"""Load the data"""

