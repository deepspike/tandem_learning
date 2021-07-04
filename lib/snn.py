import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.functional import LinearIF, Conv2dIF, Conv1dIF
import numpy as np


class LinearBN1d_if(nn.Module):
	"""Compound layer with IF neuron: Linear + BN"""

	def __init__(self, D_in, D_out, device=torch.device('cuda'), bias=True, neuronParam=None):
		super(LinearBN1d_if, self).__init__()
		self.linearif = LinearIF.apply
		self.linear = torch.nn.Linear(D_in, D_out, bias=bias)
		self.device = device
		self.bn1d = torch.nn.BatchNorm1d(D_out, eps=1e-4, momentum=0.9)
		self.neuronParam = neuronParam
		nn.init.normal_(self.bn1d.weight, 0, 2.0)

	def forward(self, input_feature_st, input_features_sc):
		# weight update based on the surrogate linear layer
		T = input_feature_st.shape[1]
		output_bn = self.bn1d(self.linear(input_features_sc))
		output = F.relu(output_bn)

		# extract the weight and bias from the surrogate linear layer
		linearif_weight = self.linear.weight.detach().to(self.device)
		linearif_bias = self.linear.bias.detach().to(self.device)

		bnGamma = self.bn1d.weight
		bnBeta = self.bn1d.bias
		bnMean = self.bn1d.running_mean
		bnVar = self.bn1d.running_var

		# re-parameterization by integrating the beta and gamma factors
		# into the 'Linear' layer weights
		ratio = torch.div(bnGamma, torch.sqrt(bnVar))
		weightNorm = torch.mul(linearif_weight.permute(1, 0), ratio).permute(1, 0) 
		biasNorm = torch.mul(linearif_bias-bnMean, ratio) + bnBeta

		# propagate the input spike train through the linearIF layer to get actual output
		# spike train
		output_st, output_sc = self.linearif(input_feature_st, output, weightNorm, self.device, biasNorm, self.neuronParam)

		return output_st, output_sc		


class ConvBN2d_if(nn.Module):
	"""Compound layer with IF neuron: Conv2d + BN"""

	def __init__(self, Cin, Cout, kernel_size, device=torch.device('cuda'), stride=1, \
					padding=0, bias=True, weight_init=2.0, neuronParam=None):

		super(ConvBN2d_if, self).__init__()
		self.conv2dIF = Conv2dIF.apply
		self.conv2d = torch.nn.Conv2d(Cin, Cout, kernel_size, stride, padding, bias=bias)
		self.bn2d = torch.nn.BatchNorm2d(Cout, eps=1e-4, momentum=0.9)
		self.device = device
		self.stride = stride
		self.padding = padding
		self.neuronParam = neuronParam
		nn.init.normal_(self.bn2d.weight, 0, weight_init)

	def forward(self, input_feature_st, input_features_sc):
		T = input_feature_st.shape[1]
		
		# weight update based on the surrogate conv2d layer
		output_bn = self.bn2d(self.conv2d(input_features_sc))
		output = F.relu(output_bn)

		# extract the weight and bias from the surrogate conv layer
		conv2d_weight = self.conv2d.weight.detach().to(self.device)
		conv2d_bias = self.conv2d.bias.detach().to(self.device)

		bnGamma = self.bn2d.weight
		bnBeta = self.bn2d.bias
		bnMean = self.bn2d.running_mean
		bnVar = self.bn2d.running_var

		# re-parameterization by integrating the beta and gamma factors
		# into the 'Conv' layer weights
		ratio = torch.div(bnGamma, torch.sqrt(bnVar))
		weightNorm = torch.mul(conv2d_weight.permute(1,2,3,0), ratio).permute(3,0,1,2) 
		biasNorm = torch.mul(conv2d_bias-bnMean, ratio) + bnBeta

		# propagate the input spike train through the IF layer to get actual output
		# spike train
		output_features_st, output_features_sc = self.conv2dIF(input_feature_st, output, weightNorm, self.device, biasNorm, self.stride, \
														self.padding, self.neuronParam)

		return output_features_st, output_features_sc							

