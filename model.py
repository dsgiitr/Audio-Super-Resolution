import numpy as numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import SubPixel1D


n_filters = [  128,  256,  512, 512, 512, 512, 512, 512]
n_filtersizes = [65, 33, 17,  9,  9,  9,  9, 9, 9]

class Down1D(nn.Module):
	"""doc string for Down1D"""
	def __init__(self, in_channel, out_channel, kernel, stride=2, padding):
		super(Down1D, self).__init__()

		self.c1 = nn.Conv1d(in_channel, out_channel, kernel_size=kernel, stride=stride, padding=kernel/2 ) 
		nn.init.orthogonal_(self.c1.weight)

	def forward(self, x):
		x1 = self.c1(x)
		x1 = F.leaky_relu(x1, negative_slope=0.2)
		return x1

class Up1D(nn.Module):
	"""doc string for Down1D"""
	def __init__(self, in_channel, out_channel, kernel, stride=2, padding):
		super(Up1D, self).__init__()

		self.c1 = nn.ConvTranspose1d(in_channel, out_channel, kernel_size=kernel, stride=stride, padding=kernel/2 )
		nn.init.orthogonal_(self.c1.weight)
		drop = nn.Dropout(p=0.5)


	def forward(self, x):
		x1 = self.c1(x)
		x1 = self.drop(x1)
		x1 = F.relu(x1)
		x1 = SubPixel1D(x, r=2)
		return x1

class Bottleneck(nn.Module):
	"""doc string for Down1D"""
	def __init__(self, in_channel, out_channel, kernel, stride=2, padding):
		super(Bottleneck, self).__init__()

		self.c1 = nn.Conv1d(in_channel, out_channel, kernel_size=kernel, stride=stride, padding=kernel/2 )
		nn.init.orthogonal_(self.c1.weight)
		drop = nn.Dropout(p=0.5)


	def forward(self, x):
		x1 = self.c1(x)
		x1 = self.drop(x1)
		x1 = F.leaky_relu(x1, negative_slope=0.2)
		return x1

class AudioUnet(nn.Module):
	def __init__(self, num_layers):
		super(AudioUnet, self).__init__()
		self.downsample = nn.ModuleList([])
		in_channels = 1
		for l, nf, fs in zip(range(num_layers), n_filters, n_filtersizes):
			self.downsample.append(Down1D(in_channels, nf, fs))
			in_channels = nf

		self.bottleneck = Bottleneck(in_channels, n_filter[-1], n_filtersizes[-1])
		
		self.upsample = nn.ModuleList([])
		for l, nf, fs in reversed(zip(range(num_layers), n_filters, n_filtersizes)):
			self.upsample.append(Up1D(in_channels, nf, fs))
			in_channels = nf

		self.final = nn.Conv1d(in_channels, 2, 9, stride=2, padding=5)
		nn.init.normal_(self.final.weight)

	def forward(self, x):
		down_outs = [x]
		for i in range(num_layers):
			down_outs.append(self.downsample[i](down_outs[i]))
		x1 = self.bottlebeck(down_outs[-1])
		for i, d in zip(range(num_layers), reversed(down_outs[1:])):
			x1 = self.upsample[i](x1)
			x1 = torch.cat([x1, down_outs[i]]) #concat axis =-1 for tf
		x1 = self.final(x1)
		x1 = SubPixel1D(x1, r=2)
		x1 = x1 + x

		return x1

