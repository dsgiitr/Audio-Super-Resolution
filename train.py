import os
import time
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import DataSet
from Model import Down1d, Up1d, Bottleneck, AudioUnet

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

	def build_model(self):
		self.model = AudioUnet(self.num_layers)
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
		train_data = DataSet(X_train, Y_train)
		val_data = DataSet(X_val, Y_val)
		


	def load(self):
		"""Not implemented"""

