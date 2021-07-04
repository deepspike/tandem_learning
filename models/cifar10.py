import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.snn import LinearBN1d_if, ConvBN2d_if
from lib.functional import InputDuplicate


class CifarNet(nn.Module):
	def __init__(self):
		super(CifarNet, self).__init__()

		self.conv1 = nn.Sequential(nn.Conv2d(3, 128, 3, stride=1, padding=1),
						nn.BatchNorm2d(128, eps=1e-4, momentum=0.9))

		self.conv2 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1),
						nn.BatchNorm2d(256, eps=1e-4, momentum=0.9))

		self.conv3 = nn.Sequential(nn.Conv2d(256, 512, 3, stride=2, padding=1),
						nn.BatchNorm2d(512, eps=1e-4, momentum=0.9))

		self.conv4 = nn.Sequential(nn.Conv2d(512, 1024, 3, stride=1, padding=1),
						nn.BatchNorm2d(1024, eps=1e-4, momentum=0.9))

		self.conv5 = nn.Sequential(nn.Conv2d(1024, 512, 3, stride=1, padding=1),
						nn.BatchNorm2d(512, eps=1e-4, momentum=0.9))

		self.fc6 = nn.Sequential(nn.Linear(8*8*512, 1024),
						nn.BatchNorm1d(1024, eps=1e-4, momentum=0.9))

		self.fc7 = nn.Linear(1024, 10)

	def forward(self, x):
		x = x.view(-1, 3, 32, 32)

		# Conv Layer
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		x = F.relu(self.conv5(x))

		# FC Layers
		x = x.view(x.size(0), -1)
		x = F.relu(self.fc6(x))
		x = self.fc7(F.dropout(x, p=0.2))

		return F.log_softmax(x, dim=1)


class CifarNetIF(nn.Module):

	def __init__(self, neuronParam, Tsim):
		super(CifarNetIF, self).__init__()
		self.T = Tsim
		self.neuronParam = neuronParam
		self.conv1 = ConvBN2d_if(3, 128, 3, stride=1, padding=1, neuronParam=self.neuronParam)
		self.conv2 = ConvBN2d_if(128, 256, 3, stride=2, padding=1, neuronParam=self.neuronParam)
		self.conv3 = ConvBN2d_if(256, 512, 3, stride=2, padding=1, neuronParam=self.neuronParam)
		self.conv4 = ConvBN2d_if(512, 1024, 3, stride=1, padding=1, neuronParam=self.neuronParam)
		self.conv5 = ConvBN2d_if(1024, 512, 3, stride=1, padding=1, neuronParam=self.neuronParam)
		self.fc5 = LinearBN1d_if(8*8*512, 1024, neuronParam=self.neuronParam)
		self.fc6 = nn.Linear(1024, 10)

	def forward(self, x):
		x = x.view(-1, 3*32*32)
		x_spike, x = InputDuplicate.apply(x, self.T)
		x_spike = x_spike.view(-1, self.T, 3, 32, 32)
		x = x.view(-1, 3, 32, 32)

		# Conv Layer
		x_spike, x = self.conv1(x_spike, x)
		x_spike, x = self.conv2(x_spike, x)
		x_spike, x = self.conv3(x_spike, x)
		x_spike, x = self.conv4(x_spike, x)
		x_spike, x = self.conv5(x_spike, x)

		# FC Layers
		x = x.view(x.size(0), -1)
		x_spike = x_spike.view(x_spike.size(0), self.T, -1)

		x_spike, x = self.fc5(x_spike, x)
		x = self.fc6(F.dropout(x, p=0.2))

		return F.log_softmax(x, dim=1)


