######################### definition of building blocks

### utils part

import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
import math

def sum_except_batch(x,num_dims=1):
	return x.reshape(*x.shape[:num_dims],-1).sum(-1)

def mean_except_batch(x,num_dims=1):
	return x.reshape(*x.shape[:num_dims],-1).mean(-1)

def loglik_nats(model,x):
	return -model.log_prob(x).mean()

def loglik_bpd(model,x):
	return -model.log_prob(x).sum() /(np.log(2)*x.shape.numel())

def elbo_nats(model,x):
	return loglik_nats(model, x)

def elbo_bpd(model,x):
	return loglik_bpd(model,x)

def iwbo(model,x,k):
	x_stack = torch.cat([x for _ in range(k)],dim=0)
	ll_stack = model.log_prob(x_stack)
	ll = torch.stack(torch.chunk(ll_stack, k, dim=0))
	return torch.logsumexp(ll,dim=0) - np.log(k)

def iwbo_batched(model, x, k, kbs):
	assert k % kbs == 0
	num_passes = k // kbs
	ll_batched = []
	for i in range(num_passes):
		x_stack = torch.cat([x for _ in range(kbs)], dim=0)
		ll_stack = model.log_prob(x_stack)
		ll_batched.append(torch.stack(torch.chunk(ll_stack, kbs, dim=0)))
	ll = torch.cat(ll_batched, dim=0)
	return torch.logsumexp(ll, dim=0) - math.log(k)

def iwbo_nats(model,x,k,kbs=None):
	if kbs: return -iwbo_batched(model,x,k,kbs).mean()
	else: return -iwbo(model,x,k).mean()	


def print_params(flow_model):
	for transform in flow_model.transforms:
		for name, param in transform.named_parameters():
			if param.requires_grad:
				print(name,param.data)
	return None


### activations part 
import torch
import torch.nn as nn
import torch.nn.functional as F



def gelu(x):
	return x*torch.sigmoid(1.702*x)

def swish(x):
	return x*torch.sigmoid(x)

def concat_relu(x):
	return F.relu(torch.cat([x,-x],dim=1))

def concat_elu(x):
	return F.elu(torch.cat([x,-x],dim=1))

def gated_tanh(x,dim):
	x_tanh, x_sigmoid = torch.chunk(x,2,dim=dim)
	return torch.tanh(x_tanh)*torch.sigmoid(x_sigmoid)

class GELU(nn.Module):

	def forward(self,input):
		return gelu(input)

class Swish(nn.Module):

	def forward(self,input):
		return swish(input)

class ConcatReLU(nn.Module):

	def forward(self,input):
		return concat_relu(input)


class GatedTanhUnit(nn.Module):

	def __init__(self,dim=1):
		super(GatedTanhUnit,self).__init__()
		self.dim = dim

	def forward(self,x):
		return gated_tanh(x,dim=self.dim)


### retrive activation layer
act_strs = {'elu','relu','gelu','swish'}
concat_act_strs = {'concat_elu','concat_relu'}

def act_module(act_str, allow_concat=False):
	if allow_concat: assert act_str in act_strs + concat_act_strs, 'Got invalid activation {}'.format(act_str)
	else: assert act_str in act_strs, 'Got invalid activation {}'.format(act_str)

	if act_str == 'relu': return nn.ReLU()
	elif act_str == 'elu': return nn.ELU()
	elif act_str == 'gelu': return GELU()
	elif act_str == 'swish': return Swish()
	elif act_str == 'concat_relu': return ConcatReLU()
	elif act_str == 'concat_elu': return ConcatELU()

def scale_fn(scale_str):
	assert scale_str in {'exp','softplus', 'sigmoid', 'tanh_exp'}
	if scale_str == 'exp': 	return lambda s: torch.exp(s)
	elif scale_str == 'softplus':	return lambda s: F.softplus(s)
	elif scale_str == 'sigmoid': return lambda s: torch.sigmoid(s+2.) + 1e-3
	elif scale_str == 'tanh_exp': return lambda s: torch.exp(2.*torch.tanh(s/2.))


# datasets part

import os
import numpy as np
import pickle
from torch.utils import data

import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader, ConcatDataset, Subset
from sklearn.model_selection import train_test_split

from torchvision.datasets import CIFAR10


class Flatten():
	def __call__(self,image):
		return image.view(-1)

class Cifar10Flatten():
	def __call__(self,image):
		return image.reshape([3*32*32])

class StaticBinarize():
	def __call__(self,image):
		return image.round().long()

class DynamicBinarize():
	def __call__(self,image):
		return image.bernoulli().long()


class Quantize():
	def __init__(self, num_bits=8):
		self.num_bits = num_bits

	def __call__(self, image):
		image = image*255
		if self.num_bits !=8:
			image = torch.floor(image/2**(8-self.num_bits))
		return image

class Dequantize():
	def __call__(self, image):
		return (image*255 + torch.rand_like(image))/256


class UnsupervisedCIFAR10(CIFAR10):
	def __init__(self,root='./',train=True,transform=None,download=False):
		super(UnsupervisedCIFAR10,self).__init__(root,train=train,transform=transform,download=download)

	def __getitem__(self,index):

		return super(UnsupervisedCIFAR10,self).__getitem__(index)[0]


class VectorizedCIFAR10():
	def __init__(self,root='./',download=True,num_bits=8,pil_transforms=[]):

		self.root = root
		trans_train = pil_transforms + [ToTensor(),Cifar10Flatten()]
		trans_test = [ToTensor(),Cifar10Flatten()]

		self.train = UnsupervisedCIFAR10(root,train=True, transform=Compose(trans_train),download=download)
		self.test = UnsupervisedCIFAR10(root,train=False, transform=Compose(trans_test))

		self.splits = ['train','test']

	@property
	def num_splits(self):
		return len(self.splits)

	def get_data_loader(self,split, batch_size, shuffle=True, pin_memory=True, num_workers=4):
		return DataLoader(getattr(self,split),batch_size=batch_size, shuffle=shuffle,pin_memory=pin_memory,num_workers=num_workers)

	def get_data_loaders(self, batch_size, shuffle=True, pin_memory=True, num_workers=4):
		data_loaders = [self.get_data_loader(split=split,batch_size=batch_size,shuffle=shuffle,
			pin_memory=pin_memory,num_workers=num_workers) for split in self.splits]
		return data_loaders

class MyCIFAR10():
	def __init__(self,root='./',download=True,num_bits=8,pil_transforms=[]):

		self.root = root
		trans_train = pil_transforms + [ToTensor(),Quantize(num_bits)]
		trans_test = [ToTensor(), Quantize(num_bits)]

		self.train = UnsupervisedCIFAR10(root,train=True, transform=Compose(trans_train),download=download)
		self.test = UnsupervisedCIFAR10(root,train=False, transform=Compose(trans_test))

		self.splits = ['train','test']

	@property
	def num_splits(self):
		return len(self.splits)

	def get_data_loader(self,split, batch_size, shuffle=True, pin_memory=True, num_workers=4):
		return DataLoader(getattr(self,split),batch_size=batch_size, shuffle=shuffle,pin_memory=pin_memory,num_workers=num_workers)

	def get_data_loaders(self, batch_size, shuffle=True, pin_memory=True, num_workers=4):
		data_loaders = [self.get_data_loader(split=split,batch_size=batch_size,shuffle=shuffle,
			pin_memory=pin_memory,num_workers=num_workers) for split in self.splits]
		return data_loaders


### transforms part

import torch
from torchvision.transforms import Compose, ToTensor
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Iterable
from functools import reduce
from operator import mul

from torch.distributions import Normal
from torch.distributions import Bernoulli
from torch.utils import checkpoint

import scipy
import copy


class LambdaLayer(nn.Module):
	def __init__(self, lambd):
		super(LambdaLayer,self).__init__()
		if lambd is None: lambd = lambda x:x
		self.lambd = lambd

	def forward(self,x):
		return self.lambd(x)

class MLP(nn.Sequential):
	def __init__(self,input_size, output_size, hidden_units, activation='relu',in_lambda=None,out_lambda=None):
		self.input_size = input_size
		self.output_size = output_size

		layers = []
		if in_lambda: layers.append(LambdaLayer(in_lambda))
		for in_size, out_size in zip([input_size]+hidden_units[:-1],hidden_units):
			layers.append(nn.Linear(in_size,out_size))
			layers.append(act_module(activation))
		layers.append(nn.Linear(hidden_units[-1], output_size))
		if out_lambda: layers.append(LambdaLayer(out_lambda))

		super(MLP, self).__init__(*layers)

class Distribution(nn.Module):

	def log_prob(self, x):

		raise NotImplementError()

	def sample(self, num_samples):

		raise NotImplementError()

	def sample_with_log_prob(self,num_samples):

		samples = self.sample(num_samples)
		log_prob = self.log_prob(samples)
		return samples, log_prob

	def forward(self, *args, mode, **kwargs):

		if mode == 'log_prob':
			return self.log_prob(*args,**kwargs)
		else:
			raise RuntimeError("Mode {} not supported.".format(mode))


class StandardNormal(Distribution):
	def __init__(self,shape):
		super(StandardNormal, self).__init__()
		self.shape = torch.Size(shape)
		self.register_buffer('buffer',torch.zeros(1))

	def log_prob(self,x):
		log_base = -0.5*np.log(2*np.pi)
		log_inner = -0.5*x**2
		return sum_except_batch(log_base+log_inner)

	def sample(self, num_samples):
		return torch.randn(num_samples,*self.shape, device=self.buffer.device, dtype=self.buffer.dtype)


class Transform(nn.Module):

	has_inverse = True

	@property
	def bijective(self):
		raise NotImplementError()

	@property
	def stochastic_forward(self):
		raise NotImplementError()

	@property
	def stochastic_inverse(self):
		raise NotImplementedError()
	@property
	def lower_bound(self):
		return self.stochastic_forward

	def forward(self,x):
		raise NotImplementError()

	def inverse(self,z):
		raise NotImplementError()


class StochasticTransform(Transform):

	has_inverse = True
	bijective = False
	stochastic_forward = True
	stochastic_inverse = True

class Bijection(Transform):

	bijective = True
	stochastic_forward = False
	stochastic_inverse = False
	lower_bound = False

class Surjection(Transform):

	bijective = False

	@property
	def stochastic_forward(self):
		raise NotImplementError()

	@property
	def stochastic_inverse(self):
		return not self.stochastic_forward

class SwitchBijection1d(Bijection):

	def forward(self,x):
		a,b = torch.chunk(x,2,1)
		z = torch.cat([b,a],dim=1)
		ldj = torch.zeros((x.shape[0],)).cuda()
		return z,ldj

	def inverse(self,z):
		a,b = torch.chunk(z,2,1)
		x = torch.cat([b,a],dim=1)
		return x


class CouplingBijection(Bijection):

	def __init__(self, coupling_net, split_dim=1, num_condition=None):
		super(CouplingBijection,self).__init__()
		assert split_dim >=1
		self.coupling_net = coupling_net
		self.split_dim = split_dim
		self.num_condition = num_condition

	def split_input(self, input):
		if self.num_condition:
			split_proportions = (self.num_condition, input.shape[self.split_dim]-self.num_condition)
			return torch.split(input, split_proportions, dim=self.split_dim)
		else:
			return torch.chunk(input, 2, dim=self.split_dim)

	def forward(self,x):

		id,x2 = self.split_input(x)
		elementwise_params = self.coupling_net(id)
		z2,ldj = self._elementwise_forward(x2, elementwise_params)
		z = torch.cat([id,z2],dim=self.split_dim)

		return z,ldj

	def inverse(self,z):
		with torch.no_grad():
			id,z2 = self.split_input(z)
			elementwise_params = self.coupling_net(id)
			x2 = self._elementwise_inverse(z2,elementwise_params)
			x = torch.cat([id,x2],dim=self.split_dim)
		return x

	def _output_dim_mutiplier(self):
		raise NotImplementError()

	def _elementwise_forward(self,x,elementwise_params):
		raise NotImplementError()

	def _elementwise_inverse(self,z,elementwise_params):
		raise NotImplementError()


class AdditiveCouplingBijection(CouplingBijection):

	def _output_dim_mutiplier(self):
		return 1

	def _elementwise_forward(self,x,elementwise_params):
		return x + elementwise_params, torch.zeros(x.shape[0],device=x.device, dtype=x.dtype)

	def _elementwise_inverse(self,z,elementwise_params):
		return z - elementwise_params

class AffineCouplingBijection(CouplingBijection):

	def __init__(self, coupling_net, split_dim=1, num_condition=None, scale_fn=lambda s:torch.exp(s)):
		super(AffineCouplingBijection,self).__init__(coupling_net=coupling_net, split_dim=split_dim,num_condition=num_condition)
		assert callable(scale_fn)
		self.scale_fn = scale_fn

	def _output_dim_mutiplier(self):
		return 2

	def _elementwise_forward(self,x, elementwise_params):
		assert elementwise_params.shape[-1] == self._output_dim_mutiplier()
		unconstrained_scale, shift = self._unconstrained_scale_and_shift(elementwise_params)
		scale = self.scale_fn(unconstrained_scale)
		z = scale*x + shift
		ldj = sum_except_batch(torch.log(scale))
		return z,ldj

	def _elementwise_inverse(self,z,elementwise_params):
		assert elementwise_params.shape[-1] == self._output_dim_mutiplier()
		unconstrained_scale, shift = self._unconstrained_scale_and_shift(elementwise_params)
		scale = self.scale_fn(unconstrained_scale)
		x = (z-shift)/scale
		return x

	def _unconstrained_scale_and_shift(self, elementwise_params):
		unconstrained_scale = elementwise_params[..., 0]
		shift = elementwise_params[...,1]
		return unconstrained_scale, shift


class AffineCouplingBijection1d(CouplingBijection):

	def __init__(self, coupling_net, split_dim=1, num_condition=None, scale_fn=lambda s:torch.exp(s),split_type='half'):
		super(AffineCouplingBijection1d,self).__init__(coupling_net=coupling_net, split_dim=split_dim,num_condition=num_condition)
		assert callable(scale_fn)
		self.scale_fn = scale_fn

		input_dim = self.coupling_net.input_size
		all_dim = int(self.coupling_net.output_size/2) + input_dim

		if split_type == 'half':
			coupling_index = input_dim
			self.coupling_index = list(np.arange(coupling_index))
		elif split_type == 'random':
			self.coupling_index = np.sort(np.random.choice(np.arange(all_dim),input_dim,replace=False))
		
		self.no_coupling_index = []
		for i in np.arange(all_dim):
			if i not in self.coupling_index:
				self.no_coupling_index.append(i)

	def _output_dim_mutiplier(self):
		return 2

	def split_input(self, input):
		if self.num_condition:
			split_proportions = (self.num_condition, input.shape[self.split_dim]-self.num_condition)
			return torch.split(input, split_proportions, dim=self.split_dim)
		else:
			if self.coupling_index is not None:
				id = input[:,self.coupling_index]
				x2 = input[:,self.no_coupling_index]
				return id,x2
			else:
				return torch.chunk(input, 2, dim=self.split_dim)

	def forward(self,x):

		id,x2 = self.split_input(x)
		elementwise_params = self.coupling_net(id)
		z2,ldj = self._elementwise_forward(x2, elementwise_params)
		z = torch.zeros(x.shape,device=x.device)
		z[:,self.coupling_index] = id
		z[:,self.no_coupling_index] = z2

		return z,ldj

	def inverse(self,z):
		with torch.no_grad():
			id,z2 = self.split_input(z)
			elementwise_params = self.coupling_net(id)
			x2 = self._elementwise_inverse(z2,elementwise_params)
			x = torch.zeros(z.shape,device=z.device)
			x[:,self.coupling_index] = id
			x[:,self.no_coupling_index] = x2

		return x

	def _elementwise_forward(self,x, elementwise_params):
		# assert elementwise_params.shape[-1] == self._output_dim_mutiplier()
		unconstrained_scale, shift = self._unconstrained_scale_and_shift(elementwise_params)
		unconstrained_scale = torch.clamp(unconstrained_scale,-2,2)
		scale = self.scale_fn(unconstrained_scale)

		z = scale*x + shift
		ldj = torch.sum(unconstrained_scale,dim=1)
		return z,ldj

	def _elementwise_inverse(self,z,elementwise_params):
		# assert elementwise_params.shape[-1] == self._output_dim_mutiplier()
		unconstrained_scale, shift = self._unconstrained_scale_and_shift(elementwise_params)
		unconstrained_scale = torch.clamp(unconstrained_scale,-2,2)

		scale = self.scale_fn(unconstrained_scale)
		x = (z-shift)/scale
		return x

	def _unconstrained_scale_and_shift(self, elementwise_params):
		unconstrained_scale,shift = torch.chunk(elementwise_params,2,self.split_dim)
		return unconstrained_scale, shift


class Flow(Distribution):

	def __init__(self, base_dist, transforms):
		super(Flow,self).__init__()
		assert isinstance(base_dist, Distribution)
		if isinstance(transforms,Transform): transforms = [transforms]
		assert isinstance(transforms, Iterable)
		assert all(isinstance(transform, Transform) for transform in transforms)
		self.base_dist = base_dist
		self.transforms = nn.ModuleList(transforms)
		self.lower_bound = any(transform.lower_bound for transform in transforms)

	def log_prob(self,x):
		log_prob = torch.zeros(x.shape[0],device=x.device)
		for transform in self.transforms:
			x,ldj = transform(x)
			log_prob += ldj

		log_prob += self.base_dist.log_prob(x)
		return log_prob


	def sample(self, num_samples):
		z = self.base_dist.sample(num_samples)
		for transform in reversed(self.transforms):
			z = transform.inverse(z)

		return z

	def sample_with_log_prob(self, num_samples):
		raise RuntimeError("Flow does not support sample_with_log_prob, see InverseFlow instead.")










########################### main body

import torch
from torchvision.transforms import Compose, ToTensor
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Iterable
from functools import reduce
from operator import mul

from torch.distributions import Normal
from torch.distributions import Bernoulli



from torch.optim import Adam

device = 'cuda' if torch.cuda.is_available() else 'cpu'


### cifar10 part

data = VectorizedCIFAR10()
train_loader, test_loader = data.get_data_loaders(64)



def net(channels):
	return nn.Sequential(DenseNet(in_channels=channels//2,
		out_channels=channels,
		num_blocks=1,
		mid_channels=64,
		depth=8,
		growth=16,
		dropout=0.0,
		gated_conv=True,
		zero_init=True),
	ElementwiseParams2d(2))

def coupling_net(input_dim):
	return MLP(int(input_dim/2), input_dim,hidden_units=[4096,2048],
                                activation='relu',
                                in_lambda=None)

# model = Flow(base_dist=StandardNormal((3072,)),
# 			transforms=[
# 			AffineCouplingBijection1d(coupling_net(3072),split_type='random'),
# 			AffineCouplingBijection1d(coupling_net(3072),split_type='random'),
# 			AffineCouplingBijection1d(coupling_net(3072),split_type='random'),
# 			AffineCouplingBijection1d(coupling_net(3072),split_type='random'),
# 			AffineCouplingBijection1d(coupling_net(3072),split_type='random'),
# 			AffineCouplingBijection1d(coupling_net(3072),split_type='random'),
# 			AffineCouplingBijection1d(coupling_net(3072),split_type='random'),
# 			AffineCouplingBijection1d(coupling_net(3072),split_type='random'),
# 			AffineCouplingBijection1d(coupling_net(3072),split_type='random')]).to(device)

model = Flow(base_dist=StandardNormal((3072,)),
			transforms=[
			AffineCouplingBijection1d(coupling_net(3072)),
			SwitchBijection1d(),AffineCouplingBijection1d(coupling_net(3072)),
			SwitchBijection1d(),AffineCouplingBijection1d(coupling_net(3072)),
			SwitchBijection1d(),AffineCouplingBijection1d(coupling_net(3072)),
			SwitchBijection1d(),AffineCouplingBijection1d(coupling_net(3072)),
			SwitchBijection1d(),AffineCouplingBijection1d(coupling_net(3072)),
			SwitchBijection1d(),AffineCouplingBijection1d(coupling_net(3072)),
			SwitchBijection1d(),AffineCouplingBijection1d(coupling_net(3072)),
			SwitchBijection1d(),AffineCouplingBijection1d(coupling_net(3072))]).to(device)



import math

epoch = 0

optimizer = Adam(model.parameters(), lr=1e-4)


print('Training...')
for epoch in range(epoch,250):
    l = 0.0
    for i, x in enumerate(train_loader):
        optimizer.zero_grad()
        loss = -model.log_prob(x.to(device)).sum() / (math.log(2) * x.numel())
        loss.backward()
        optimizer.step()
        l += loss.detach().cpu().item()
        print('Epoch: {}/{}, Iter: {}/{}, Bits/dim: {:.3f}'.format(epoch+1, 10, i+1, len(train_loader), l/(i+1)))
    print('')




##########
## Test ##
##########

print('Testing...')
with torch.no_grad():
    l = 0.0
    for i, x in enumerate(test_loader):
        loss = -model.log_prob(x.to(device)).sum() / (math.log(2) * x.numel())
        l += loss.detach().cpu().item()
        print('Iter: {}/{}, Bits/dim: {:.3f}'.format(i+1, len(test_loader), l/(i+1)))
    print('')

############
## Sample ##
############

print('Sampling...')
img = torch.from_numpy(data.test.data[:64]).permute([0,3,1,2])
with torch.no_grad():
	samples = model.sample(64)
samples = torch.floor(255*torch.clamp(samples,0.,1.))
vutils.save_image(img.view(img.shape[0],3,32,32).cpu().float()/255, fp='cifar10_data.png', nrow=8)


vutils.save_image(samples.view(samples.shape[0],3,32,32).cpu().float()/255, fp='cifar10_vec_flow.png', nrow=8)

