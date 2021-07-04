import torch
import torch.nn.functional as F


def LIF(x, mem, spike, vthr, leaky_rate):
	""" leaky integrate-and-fire Neuron Model """

	mem = mem * leaky_rate + x - vthr * spike
	spike = torch.ge(mem, vthr).float()

	return mem, spike


def IF(x, mem, spike, vthr):
	""" integrate-and-fire Neuron Model """

	mem = mem + x - spike*vthr
	spike = torch.ge(mem, vthr).float()
	
	return mem, spike


class InputDuplicate(torch.autograd.Function):
	"""
    Utility class for duplicating the real-valued inputs (as the input current) over the time window T
    """
	@staticmethod
	def forward(ctx, input_image, T):
		"""
		Params:
			input_image: normalized within (0,1)
			T: simulation time widow size
		Returns:
			input_image_distribute: duplicated input images that distribute over the time window
			input_image_aggregate: aggregated input images
		"""
		input_image_distribute = input_image.unsqueeze(dim=1).repeat(1, T, 1)
		input_image_aggregate = input_image_distribute.sum(dim=1)

		return input_image_distribute, input_image_aggregate

	@staticmethod
	def backward(ctx, grad_spike_out, grad_spike_count_out):
		"""Auxiliary function only, no gradient required"""

		return None, None


class LinearIF(torch.autograd.Function):
	"""Fully-connected SNN layer"""
	@staticmethod
	def forward(ctx, spike_in, ann_output, weight, device=torch.device('cuda'), bias=None, neuronParam=None):
		"""
		Params:
			spike_in: input spike trains 
			ann_output: placeholder
			weight: connection weights
			device: cpu or gpu
			bias: neuronal bias parameters
			neuronParam： neuronal parameters
		Returns:
			spike_out: output spike trains
			spike_count_out: output spike counts
		"""
		supported_neuron = {
			'IF': IF,
			'LIF': LIF,
		}
		if neuronParam['neuronType'] not in supported_neuron:
			raise RuntimeError("Unsupported Neuron Model: {}".format(neuronParam['neuronType']))
		N, T, _ = spike_in.shape
		out_features = bias.shape[0]
		pot_in = spike_in.matmul(weight.t())
		spike_out = torch.zeros_like(pot_in, device=device)
		bias_distribute = bias.repeat(N, 1)/T # distribute bias through simulation time steps
		mem = torch.zeros(N, out_features, device=device)
		spike = torch.zeros(N, out_features, device=device) # init input spike train

		# Iterate over simulation time steps to determine output spike trains
		for t in range(T):
			x = pot_in[:,t,:].squeeze() + bias_distribute
			# Membrane potential update
			if neuronParam['neuronType'] == 'IF':
				mem, spike = IF(x, mem, spike, neuronParam['vthr'])
			elif neuronParam['neuronType'] == 'LIF':
				mem, spike = LIF(x, mem, spike, neuronParam['vthr'], neuronParam['leaky_rate_mem'])
			spike_out[:,t,:] = spike

		spike_count_out = torch.sum(spike_out, dim=1).squeeze(dim=1)

		return spike_out, spike_count_out

	@staticmethod
	def backward(ctx, grad_spike_out, grad_spike_count_out):
		"""Auxiliary function only, no gradient required"""
		grad_ann_out = grad_spike_count_out.clone()

		return None, grad_ann_out, None, None, None, None


class Conv1dIF(torch.autograd.Function):
	"""1D Convolutional Layer"""
	@staticmethod
	def forward(ctx, spike_in, features_in, weight, device=torch.device('cuda'), bias=None,\
					stride=1, padding=0, dilation=1, neuronParam=None):
		"""
		Params:
			spike_in: input spike trains 
			features_in: placeholder
			weight: connection weights
			device: cpu or gpu
			bias: neuronal bias parameters
			stride: stride of 1D Conv
			padding: padding of 1D Conv 
			dilation: dilation of 1D Conv 
			neuronParam： neuronal parameters
		Returns:
			spike_out: output spike trains
			spike_count_out: output spike counts
		"""
		supported_neuron = {
			'IF': IF,
			'LIF': LIF,
		}
		if neuronParam['neuronType'] not in supported_neuron:
			raise RuntimeError("Unsupported Neuron Model: {}".format(neuronParam['neuronType']))
		N, T, in_channels, iW = spike_in.shape
		out_channels, in_channels, kW = weight.shape
		mem = torch.zeros_like(F.conv1d(spike_in[:,0,:,:], weight, bias, stride, padding, dilation))
		bias_distribute = F.conv1d(torch.zeros_like(spike_in[:,0,:,:]), weight, bias, stride, padding, dilation)/T # distribute bias through simulation time steps
		_, _, outW = mem.shape
		spike_out = torch.zeros(N, T, out_channels, outW, device=device)
		spike = torch.zeros(N, out_channels, outW, device=device) # init input spike train

		# Iterate over simulation time steps to determine output spike trains
		for t in range(T):
			x = F.conv1d(spike_in[:,t,:,:], weight, None, stride, padding, dilation) + bias_distribute
			# Membrane potential update
			if neuronParam['neuronType'] == 'IF':
				mem, spike = IF(x, mem, spike, neuronParam['vthr'])
			elif neuronParam['neuronType'] == 'LIF':
				mem, spike = LIF(x, mem, spike, neuronParam['vthr'], neuronParam['leaky_rate_mem'])

			spike_out[:,t,:,:] = spike

		spike_count_out = torch.sum(spike_out, dim=1)

		return spike_out, spike_count_out		

	@staticmethod
	def backward(ctx, grad_spike_out, grad_spike_count_out):
		"""Auxiliary function only, no gradient required"""

		grad_spike_count_out = grad_spike_count_out.clone()
		grad_spike_in, grad_weight, grad_device, grad_bias, grad_stride, grad_padding, grad_dilation = None, \
				None, None, None, None, None, None

		return None, grad_spike_count_out, None, None, None, None, None, None, None


class Conv2dIF(torch.autograd.Function):
	"""2D Convolutional Layer"""
	@staticmethod
	def forward(ctx, spike_in, features_in, weight, device=torch.device('cuda'), bias=None,\
					stride=1, padding=0, neuronParam=None):
		"""
		Params:
			spike_in: input spike trains 
			features_in: placeholder
			weight: connection weights
			device: cpu or gpu
			bias: neuronal bias parameters
			stride: stride of 1D Conv
			padding: padding of 1D Conv 
			dilation: dilation of 1D Conv 
			neuronParam： neuronal parameters
		Returns:
			spike_out: output spike trains
			spike_count_out: output spike counts
		"""
		supported_neuron = {
			'IF': IF,
			'LIF': LIF,
		}
		if neuronParam['neuronType'] not in supported_neuron:
			raise RuntimeError("Unsupported Neuron Model: {}".format(neuronParam['neuronType']))
		N, T, in_channels, iH, iW = spike_in.shape
		out_channels, in_channels, kH, kW = weight.shape
		mem = torch.zeros_like(F.conv2d(spike_in[:,0,:,:,:], weight, bias, stride, padding))
		bias_distribute = F.conv2d(torch.zeros_like(spike_in[:,0,:,:,:]), weight, bias, stride, padding)/T # init the membrane potential with the bias
		_, _, outH, outW = mem.shape
		spike_out = torch.zeros(N, T, out_channels, outH, outW, device=device)
		spike = torch.zeros(N, out_channels, outH, outW, device=device) # init input spike train

		# Iterate over simulation time steps to determine output spike trains
		for t in range(T):
			x = F.conv2d(spike_in[:,t,:,:,:], weight, None, stride, padding) + bias_distribute
			# Membrane potential update
			if neuronParam['neuronType'] == 'IF':
				mem, spike = IF(x, mem, spike, neuronParam['vthr'])
			elif neuronParam['neuronType'] == 'LIF':
				mem, spike = LIF(x, mem, spike, neuronParam['vthr'], neuronParam['leaky_rate_mem'])

			spike_out[:,t,:,:] = spike

		spike_count_out = torch.sum(spike_out, dim=1)

		return spike_out, spike_count_out		

	@staticmethod
	def backward(ctx, grad_spike_out, grad_spike_count_out):
		"""Auxiliary function only, no gradient required"""

		grad_spike_count_out = grad_spike_count_out.clone()
		grad_spike_in, grad_weight, grad_device, grad_bias, grad_stride, grad_padding = None, \
				None, None, None, None, None

		return None, grad_spike_count_out, None, None, None, None, None, None


