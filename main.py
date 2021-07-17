import os
import torch

from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader, ConcatDataset, Subset
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

# data part 


class UnsupervisedMNIST(MNIST):
	def __init__(self,root='./',train=True,transform=None,download=False):
		super(UnsupervisedMNIST,self).__init__(root,train=train,transform=transform,download=download)

	def __getitem__(self,index):

		return super(UnsupervisedMNIST, self).__getitem__(index)[0]

	@property
	def raw_folder(self):
		return os.path.join(self.root, 'MNIST','raw')
	@property
	def processed_folder(self):
		return os.path.join(self.root,'MNIST','processed')

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
		return image.long()

# DataSet = UnsupervisedMNIST(download=True)

# plt.imshow(DataSet[0]);plt.show()


# ToTensorConverter = ToTensor()
# StaticBinarizeConverter = StaticBinarize()
# DynamicBinarizeConverter = DynamicBinarize()
# FlattenConverter = Flatten()
# QuantizeConverter = Quantize()



# image = ToTensorConverter(DataSet[0])
# StaticBinarizeConverter(image)
# DynamicBinarizeConverter(image)
# plt.imshow(image.squeeze(),cmap='gray');plt.show()
# plt.imshow(StaticBinarizeConverter(image).squeeze(),cmap='gray');plt.show()
# plt.imshow(DynamicBinarizeConverter(image).squeeze(),cmap='gray');plt.show()
# plt.imshow(QuantizeConverter(image).squeeze(),cmap='gray');plt.show()


# class PresplitLoader():

# 	@property
# 	def num_splits(self):
# 		return len(self.splits)

# 	def get_data_loader(self,split, batch_size, shuffle=True, pin_memory=True, num_workers=4):
# 		return DataLoader(getattr(self,split),batch_size=batch_size, shuffle=shuffle,pin_memory=pin_memory,num_workers=num_workers)

# 	def get_data_loaders(self, batch_size, shuffle=True, pin_memory=True, num_workers=4):
# 		data_loaders = [self.get_data_loader(split=split,batch_size=batch_size,shuffle=shuffle,
# 			pin_memory=pin_memory,num_workers=num_workers) for split in self.splits]
# 		return data_loaders


# class TrainTestLoader(PresplitLoader):
# 	splits = ['train','test']

# class TrainValidTestLoader(PresplitLoader):
# 	splits = ['train','valid','test']

# 	@property
# 	def train_valid(self):
# 		return ConcatDataset([self.train,self.valid])

	


class DynamicallyBinarizedMNIST():

	def __init__(self,root='./',download=True,flatten=False):

		self.root = root

		trans = [ToTensor(),DynamicBinarize()]
		if flatten: trans.append(Flatten())

		self.train = UnsupervisedMNIST(root,train=True, transform=Compose(trans), download=download)
		self.test = UnsupervisedMNIST(root,train=False, transform=Compose(trans))

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


from torchvision.datasets import CIFAR10
class UnsupervisedCIFAR10(CIFAR10):
	def __init__(self,root='./',train=True,transform=None,download=False):
		super(UnsupervisedCIFAR10,self).__init__(root,train=train,transform=transform,download=download)

	def __getitem__(self,index):

		return super(UnsupervisedCIFAR10,self).__getitem__(index)[0]


class VectorizedCIFAR10():
	def __init__(self,root='./',download=True,num_bits=8,pil_transforms=[]):

		self.root = root
		trans_train = pil_transforms + [ToTensor(),Quantize(num_bits),Cifar10Flatten()]
		trans_test = [ToTensor(), Quantize(num_bits),Cifar10Flatten()]

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



# DataSets = DynamicallyBinarizedMNIST(root='./',download=True,flatten=False)

# train_loader, test_loader = DataSets.get_data_loaders(batch_size=128,shuffle=True,pin_memory=True,num_workers=4) 


class LambdaLayer(nn.Module):
	def __init__(self, lambd):
		super(LambdaLayer,self).__init__()
		if lambd is None: lambd = lambda x:x
		self.lambd = lambd

	def forward(self,x):
		return self.lambd(x)


class Flatten(nn.Module):
	def forward(self,x):
		return x.view(x.shape[0],-1)



### activations 

def gelu(x):
	return x.torch.sigmoid(1.702*x)

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


class MLP(nn.Sequential):
	def __init__(self,input_size, output_size, hidden_units, activation='relu',in_lambda=None,out_lambda=None):
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



import numpy as np

def sum_except_batch(x,num_dims=1):
	return x.reshape(*x.shape[:num_dims],-1).sum(-1)

def mean_except_batch(x,num_dims=1):
	return x.reshape(*x.shape[:num_dims],-1).mean(-1)

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

class DiagonalNormal(Distribution):

	def __init__(self, shape):
		super(DiagonalNormal, self).__init__()
		self.shape = torch.Size(shape)
		self.loc = nn.Parameter(torch.zeros.shape)
		self.log_scale = nn.Parameter(torch.zeros(shape))

	def log_prob(self,x):
		log_base = -0.5*torch.log(2*np.pi) - self.log_scale
		log_inner = -0.5*torch.exp(-2*self.log_scale)*((x-self.loc)**2)
		return sum_except_batch(log_base + log_inner)

	def sample(self,x):
		eps = torch.randn(num_samples, *self.shape, device=self.loc.device, dtype=self.loc.dtype)
		return self.loc + self.log_scale.exp()*eps

class ConvNormal2d(DiagonalNormal):
	def __init__(self, shape):
		super(DiagonalNormal, self).__init__()
		assert len(shape) ==3
		self.shape = torch.Size(shape)
		self.loc = torch.nn.Parameter(torch.zeros(1,shape[0],1,1))
		self.log_scale = torch.nn.Parameter(torch.zeros(1,shape[0],1,1))

class ConditionalDistribution(Distribution):

	def log_prob(self,x,context):
		raise NotImplementError()
	def sample(self,context):
		raise NotImplementError()
	def sample_with_log_prob(self, context):
		raise NotImplementError()

from torch.distributions import Normal

class ConditionalMeanNormal(ConditionalDistribution):
    """A multivariate Normal with conditional mean and fixed std."""

    def __init__(self, net, scale=1.0):
        super(ConditionalMeanNormal, self).__init__()
        self.net = net
        self.scale = scale

    def cond_dist(self, context):
        mean = self.net(context)
        return Normal(loc=mean, scale=self.scale)

    def log_prob(self, x, context):
        dist = self.cond_dist(context)
        return sum_except_batch(dist.log_prob(x))

    def sample(self, context):
        dist = self.cond_dist(context)
        return dist.rsample()

    def sample_with_log_prob(self, context):
        dist = self.cond_dist(context)
        z = dist.rsample()
        log_prob = dist.log_prob(z)
        log_prob = sum_except_batch(log_prob)
        return z, log_prob

    def mean(self, context):
        return self.cond_dist(context).mean


class ConditionalMeanStdNormal(ConditionalDistribution):
    """A multivariate Normal with conditional mean and learned std."""

    def __init__(self, net, scale_shape):
        super(ConditionalMeanStdNormal, self).__init__()
        self.net = net
        self.log_scale = nn.Parameter(torch.zeros(scale_shape))

    def cond_dist(self, context):
        mean = self.net(context)
        return Normal(loc=mean, scale=self.log_scale.exp())

    def log_prob(self, x, context):
        dist = self.cond_dist(context)
        return sum_except_batch(dist.log_prob(x))

    def sample(self, context):
        dist = self.cond_dist(context)
        return dist.rsample()

    def sample_with_log_prob(self, context):
        dist = self.cond_dist(context)
        z = dist.rsample()
        log_prob = dist.log_prob(z)
        log_prob = sum_except_batch(log_prob)
        return z, log_prob

    def mean(self, context):
        return self.cond_dist(context).mean


class ConditionalNormal(ConditionalDistribution):
    """A multivariate Normal with conditional mean and log_std."""

    def __init__(self, net, split_dim=-1):
        super(ConditionalNormal, self).__init__()
        self.net = net
        self.split_dim = split_dim

    def cond_dist(self, context):
        params = self.net(context)
        mean, log_std = torch.chunk(params, chunks=2, dim=self.split_dim)
        return Normal(loc=mean, scale=log_std.exp())

    def log_prob(self, x, context):
        dist = self.cond_dist(context)
        return sum_except_batch(dist.log_prob(x))

    def sample(self, context):
        dist = self.cond_dist(context)
        return dist.rsample()

    def sample_with_log_prob(self, context):
        dist = self.cond_dist(context)
        z = dist.rsample()
        log_prob = dist.log_prob(z)
        log_prob = sum_except_batch(log_prob)
        return z, log_prob

    def mean(self, context):
        return self.cond_dist(context).mean

    def mean_stddev(self, context):
        dist = self.cond_dist(context)
        return dist.mean, dist.stddev


encoder.cond_dist(x)

from torch.distributions import Bernoulli


class ConditionalBernoulli(ConditionalDistribution):
    """A Bernoulli distribution with conditional logits."""

    def __init__(self, net):
        super(ConditionalBernoulli, self).__init__()
        self.net = net

    def cond_dist(self, context):
        logits = self.net(context)
        return Bernoulli(logits=logits)

    def log_prob(self, x, context):
        dist = self.cond_dist(context)
        return sum_except_batch(dist.log_prob(x.float()))

    def sample(self, context):
        dist = self.cond_dist(context)
        return dist.sample().long()

    def sample_with_log_prob(self, context):
        dist = self.cond_dist(context)
        z = dist.sample()
        log_prob = dist.log_prob(z)
        log_prob = sum_except_batch(log_prob)
        return z.long(), log_prob

    def logits(self, context):
        return self.cond_dist(context).logits

    def probs(self, context):
        return self.cond_dist(context).probs

    def mean(self, context):
        return self.cond_dist(context).mean

    def mode(self, context):
        return (self.cond_dist(context).logits>=0).long()



from collections.abc import Iterable

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

class VAE(StochasticTransform):

	def __init__(self, decoder, encoder):
		super(VAE,self).__init__()
		self.decoder = decoder
		self.encoder = encoder

	def forward(self,x):
		z, log_qz = self.encoder.sample_with_log_prob(context=x)
		log_px = self.decoder.log_prob(x,context=z)
		return z,log_px-log_qz

	def inverse(self,z):
		return self.decoder.sample(context=z)

class FlattenTransform(Transform):

	def __init__(self,in_shape):
		super(FlattenTransform,self).__init__()
		self.trans = Flatten()
		self.in_shape = in_shape

		has_inverse = True
		bijective = True
		stochastic_forward = False
		stochastic_inverse = False

	def forward(self,x):
		return self.trans(x)

	def inverse(self,x): 
		return x.view(self.in_shape)

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



x,ldj = transform(x.to(device))

encoder.sample_with_log_prob(context=x)

VAE(encoder=encoder, decoder=decoder)(x)














from torch.optim import Adam

device = 'cuda' if torch.cuda.is_available() else 'cpu'






### mnist part
DataSets = DynamicallyBinarizedMNIST()
train_loader, test_loader = DataSets.get_data_loaders(128)



LATENT_DIM = 50
encoder = ConditionalNormal(MLP(784, 2*LATENT_DIM,hidden_units=[512,256],
                                activation='relu',
                                in_lambda=lambda x: 2 * x.view(x.shape[0], 784).float() - 1))


decoder = ConditionalBernoulli(MLP(LATENT_DIM, 784,
                                   hidden_units=[512,256],
                                   activation='relu',
                                   out_lambda=lambda x: x.view(x.shape[0], 1, 28, 28)))


model = Flow(base_dist=StandardNormal((LATENT_DIM,)),
             transforms=[
                VAE(encoder=encoder, decoder=decoder)
             ]).to(device)


optimizer = Adam(model.parameters(), lr=1e-3)

for epoch in range(50):
	l = 0.0
	for i,x in enumerate(train_loader):
		optimizer.zero_grad()
		loss = -model.log_prob(x.to(device)).mean()
		loss.backward()
		optimizer.step()
		l += loss.detach().cpu().item()
		print('Epoch: {}/{}, Iter: {}/{}, Nats: {:.3f}'.format(epoch+1,20,i+1,len(train_loader),l/(i+1)),end='\r')
	print('')


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

with torch.no_grad():
	l = 0.0
	for i,x in enumerate(test_loader):
		loss = iwbo_nats(model, x.to(device), k=10)
		l += loss.detach().cpu().item()
		print('Iter: {}/{}, Nats: {:.3f}'.format(i+1, len(test_loader), l/(i+1)), end='\r')
	print('')



img = next(iter(test_loader))[:64]
samples = model.sample(64)

import torchvision.utils as vutils


vutils.save_image(img.cpu().float(), fp='mnist_data.png', nrow=8)
vutils.save_image(samples.cpu().float(), fp='mnist_vae.png', nrow=8)













############################################## image part

class Surjection(Transform):

	bijective = False

	@property
	def stochastic_forward(self):
		raise NotImplementError()

	@property
	def stochastic_inverse(self):
		return not self.stochastic_forward


class UniformDequantization(Surjection):

	stochastic_forward = True

	def __init__(self, num_bits=8):
		super(UniformDequantization, self).__init__()
		self.num_bits = num_bits
		self.quantization_bins = 2**num_bits
		self.register_buffer('ldj_per_dim',-torch.log(torch.tensor(self.quantization_bins, dtype=torch.float)))

	def _ldj(self, shape):
		batch_size = shape[0]
		num_dims = shape[1:].numel()
		ldj = self.ldj_per_dim*num_dims

		return ldj.repeat(batch_size)


	def forward(self,x):
		u = torch.randn(x.shape,device=self.ldj_per_dim.device,dtype=self.ldj_per_dim.dtype)
		z = (x.type(u.dtype) + u)/self.quantization_bins
		ldj = self._ldj(z.shape)
		return z,ldj

	def inverse(self,z):
		z = self.quantization_bins*z
		return z.floor().clamp(min=0,max=self.quantization_bins-1).long()



class Bijection(Transform):

	bijective = True
	stochastic_forward = False
	stochastic_inverse = False
	lower_bound = False

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

class _ActNormBijection(Bijection):

	def __init__(self, num_features, data_dep_init=True, eps=1e-6):
		super(_ActNormBijection,self).__init__()
		self.num_features = num_features
		self.data_dep_init = data_dep_init
		self.eps = eps

		self.register_buffer('initialized',torch.zeros(1) if data_dep_init else torch.ones(1))
		self.register_params()

	def data_init(self,x):
		self.initialized += 1.
		with torch.no_grad():
			x_mean, x_std = self.compute_stats(x)
			self.shift.data = x_mean
			self.log_scale.data = torch.log(x_std + self.eps)

	def forward(self,x):
		if self.training and not self.initialized: self.data_init(x)
		z = (x - self.shift)*torch.exp(-self.log_scale)
		ldj = torch.sum(-self.log_scale).expand([x.shape[0]])*self.ldj_multiplier(x)
		return z,ldj

	def inverse(self,z):
		return self.shift + z*torch.exp(self.log_scale)

	def register_params(self):
		raise NotImplementError()

	def compute_stats(self,x):
		raise NotImplementError()

	def ldj_multiplier(self,x):
		raise NotImplementError()


class ActNormBijection(_ActNormBijection):

	def register_params(self):

		self.register_parameter('shift',nn.Parameter(torch.zeros(1,self.num_features)))
		self.register_parameter('log_scale',nn.Parameter(torch.zeros(1,self.num_features)))

	def compute_stats(self,x):

		x_mean = torch.mean(x, dim=0, keepdim=True)
		x_std = torch.std(x, dim=0, keepdim=True)

		return x_mean, x_std

	def ldj_multiplier(self,x):

		return 1

class ActNormBijection1d(_ActNormBijection):

	def register_params(self):
		self.register_parameter('shift', nn.Parameter(torch.zeros(1,self.num_features,1)))
		self.register_parameter('log_scale', nn.Parameter(torch.zeros(1, self.num_features,1)))

	def compute_stats(self,x):

		x_mean = torch.mean(x, dim=[0,2], keepdim=True)
		x_std = torch.std(x, dim=[0,2], keepdim=True)

		return x_mean, x_std

	def ldj_multiplier(self,x):
		return x.shape[2]


class ActNormBijection2d(_ActNormBijection):

	def register_params(self):

		self.register_parameter('shift',nn.Parameter(torch.zeros(1,self.num_features,1,1)))
		self.register_parameter('log_scale',nn.Parameter(torch.zeros(1,self.num_features, 1,1)))
	
	def compute_stats(self,x):

		x_mean = torch.mean(x,dim=[0,2,3],keepdim=True)
		x_std = torch.std(x,dim=[0,2,3],keepdim=True)

		return x_mean, x_std

	def ldj_multiplier(self,x):
		return x.shape[2:4].numel()


from functools import reduce
from operator import mul

class Conv1x1(Bijection):

	def __init__(self, num_channels, orthogonal_init=True, slogdet_cpu=True):
		super(Conv1x1, self).__init__()

		self.num_channels = num_channels
		self.slogdet_cpu = slogdet_cpu
		self.weight = nn.Parameter(torch.Tensor(num_channels,num_channels))
		self.reset_parameters(orthogonal_init)

	def reset_parameters(self, orthogonal_init):

		self.orthogonal_init = orthogonal_init

		if self.orthogonal_init:
			nn.init.orthogonal_(self.weight)
		else:
			bound = 1.0/ np.sqrt(self.num_channels)
			nn.init.uniform_(self.weight, -bound, bound)

	def _conv(self,weight, v):
		_,channel, *features = v.shape
		n_feature_dims = len(features)

		fill = (1,)*n_feature_dims
		weight = weight.view(channel, channel, *fill)

		if n_feature_dims == 1:
			return F.conv1d(v,weight)
		elif n_feature_dims == 2:
			return F.conv2d(v,weight)
		elif n_feature_dims == 3:
			return F.conv3d(v,weight)
		else:
			raise ValueError(f'Got {n_feature_dims}d tensor, expected 1d, 2d, or 3d')

	def _logdet(self, x_shape):
		b,c,*dims = x_shape
		if self.slogdet_cpu:
			_, ldj_per_pixel = torch.slogdet(self.weight.to('cpu'))
		else:
			_,ldj_per_pixel = torch.slogdet(self.weight)
		ldj = ldj_per_pixel * reduce(mul, dims)
		return ldj.expand([b]).to(self.weight.device)

	def forward(self,x):
		z = self._conv(self.weight,x)
		ldj = self._logdet(x.shape)

		return z,ldj

	def inverse(self,z):
		weight_inv = torch.inverse(self.weight)
		x = self._conv(weight_inv, z)
		return x


x = torch.randn([64,3,32,32])
transform = Conv1x1(3)
transform(x)


class StandardUniform(Distribution):
	def __init__(self, shape):
		super().__init__()
		self.shape = torch.Size(shape)
		self.register_buffer('zero',torch.zeros(1))
		self.register_buffer('one',torch.ones(1))

	def log_prob(self,x):
		lb = mean_except_batch(x.ge(self.zero).type(self.zero.dtype))
		ub = mean_except_batch(x.le(self.one).type(self.one.dtype))
		return torch.log(lb*ub)

	def sample(self, num_samples):
		return torch.rand((num_samples,)+self.shape,device=self.zero.device, dtype=self.zero.dtype)



class Augment(Surjection):

	stochastic_forward = True

	def __init__(self, encoder, x_size, split_dim=1):
		super(Augment, self).__init__()
		assert split_dim >= 1
		self.encoder = encoder
		self.split_dim = split_dim
		self.x_size = x_size
		self.cond = isinstance(self.encoder, ConditionalDistribution)

	def split_z(self,z):
		split_proportions = (self.x_size, z.shape[self.split_dim]-self.x_size)
		return torch.split(z, split_proportions, dim=self.split_dim)

	def forward(self,x):
		if self.cond: z2, logqz2 = self.encoder.sample_with_log_prob(context=x)
		else: z2,logqz2=self.encoder.sample_with_log_prob(num_samples=x.shape[0])

		z = torch.cat([x,z2],dim=self.split_dim)
		ldj = -logqz2
		return z,ldj 

	def inverse(self, z):
		x, z2 = self.split_z(z)
		return x


x = torch.randn([64,3,32,32])
transform = Augment(StandardUniform((3,32,32)),x_size=3)
transform(x)


class Slice(Surjection):

	stochastic_forward = False

	def __init__(self, decoder, num_keep, dim=1):
		super(Slice, self).__init__()
		assert dim >= 1
		self.decoder = decoder
		self.dim = dim 
		self.num_keep = num_keep
		self.cond = isinstance(self.decoder, ConditionalDistribution)

	def split_input(self, input):
		split_proportions = (self.num_keep, input.shape[self.dim]-self.num_keep)
		return torch.split(input, split_proportions, dim=self.dim)

	def forward(self,x):
		z, x2 = self.split_input(x)
		if self.cond: ldj = self.decoder.log_prob(x2, context=z)
		else: ldj = self.decoder.log_prob(x2)
		return z, ldj

	def inverse(self,z):
		if self.cond: x2 = self.decoder.sample(context=z)
		else: x2 = self.decoder.sample(num_samples=z.shape[0])
		x = torch.cat([z,x2],dim=self.dim)
		return x



class Squeeze2d(Bijection):

	def __init__(self,factor=2, ordered=False):
		super(Squeeze2d,self).__init__()
		assert isinstance(factor, int)
		assert factor >1
		self.factor = factor
		self.ordered = ordered

	def _squeeze(self,x):
		assert len(x.shape) == 4, 'Dimension should be 4, but was {}'.format(len(x.shape))
		batch_size,c,h,w = x.shape
		assert h % self.factor == 0, 'h = {} not multiplicative of {}'.format(h, self.factor)
		assert w % self.factor == 0, 'w = {} not multiplicative of {}'.format(w, self.factor)
		t = x.view(batch_size, c, h//self.factor, self.factor, w//self.factor, self.factor)
		if not self.ordered:
			t = t.permute(0,1,3,5,2,4).contiguous()
		else:
			t = t.permute(0,3,5,1,2,4).contiguous()

		z = t.view(batch_size, c*self.factor**2, h//self.factor, w//self.factor)
		return z


	def _unsqueeze(self, z):
		assert len(z.shape) == 4, 'Dimension should be 4, but was {}'.format(len(x.shape))
		batch_size,c,h,w = z.shape
		assert c % (self.factor ** 2) == 0, 'c = {} not multiplicative of {}'.format(c, self.factor ** 2)
		if not self.ordered:
			t = z.view(batch_size,c//self.factor**2, self.factor, self.factor,h,w)
			t = t.permute(0,1,4,2,5,3).contiguous()
		else:
			t = z.view(batch_size,self.factor, self.factor, c//self.factor**2, h,w)
			t = t.permute(0,3,4,1,5,2).contiguous()

		x = t.view(batch_size, c//self.factor**2, h*self.factor,w*self.factor)
		return x

	def forward(self,x):
		z = self._squeeze(x)
		ldj = torch.zeros(x.shape[0],device=x.device,dtype=x.dtype)
		return z,ldj

	def inverse(self,z):
		x = self._unsqueeze(z)
		return x



x = torch.tensor(
	[[[[1,2,5,6],
	[3,4,7,8],
	[10,20,50,60],
	[30,40,50,60]]]]
	)

transform = Squeeze2d(ordered=True)
transform(x)


class ElementwiseParams(nn.Module):

	def __init__(self, num_params, mode='interleaved'):
		super(ElementwiseParams, self).__init__()
		assert model in {'interleaved','sequential'}
		self.num_params = num_params
		self.mode = mode
	
	def forward(self,x):
		assert x.dim()==2, 'Expected input of shape (B,D)'
		if self.num_params !=1:
			assert x.shape[1] % self.num_params == 0
			dims = x.shape[1] //self.num_params

			if self.mode == 'interleaved':
				x = x.reshape(x.shape[0:1] + (self.num_params,dims))
				x = x.permute([0,2,1])

			elif self.mode == 'sequential':
				x = x.reshape(x.shape[0:1] + (dims, self.num_params))

		return x

class ElementwiseParams1d(nn.Module):

	def __init__(self, num_params, mode='interleaved'):
		super(ElementwiseParams1d,self).__init__()
		assert mode in {'interleaved', 'sequential'}
		self.num_params = num_params
		self.mode = mode

	def forward(self,x):
		assert x.dim()==3, 'Expected input of shape (B,D,L)'
		if self.num_params != 1:
			assert x.shape[1] % self.num_params ==0
			dims = x.shape[1] //self.num_params

			if self.mode == 'interleaved':
				x = x.reshape(x.shape[0:1] + (self.num_params,dims)+ x.shape[2:])

				x = x.permute([0,2,3,1])

			elif self.mode == 'sequential':
				x = x.reshape(x.shape[0:1] + (dims, self.num_params) + x.shape[2:])
				x = x.permute([0,1,3,2])

		return x

class ElementwiseParams2d(nn.Module):

	def __init__(self, num_params, mode='interleaved'):
		super(ElementwiseParams2d, self).__init__()
		assert mode in {'interleaved', 'sequential'}
		self.num_params = num_params
		self.mode = mode

	def forward(self, x):
		assert x.dim() == 4, 'Expected input of shape (B,C,H,W)'
		if self.num_params != 1:
			assert x.shape[1] % self.num_params == 0
			channels = x.shape[1] // self.num_params
			if self.mode == 'interleaved':
				x = x.reshape(x.shape[0:1] + (self.num_params, channels) + x.shape[2:])
				x = x.permute([0,2,3,4,1])
			elif self.mode == 'sequential':
				x = x.reshape(x.shape[0:1] + (channels, self.num_params) + x.shape[2:])
				x = x.permute([0,1,3,4,2])
		return x



x = torch.randn([64,6,32,32])

mapping = ElementwiseParams2d(2)

num_params = 2
mapping(x)



class DensLayer(nn.Module):
	def __init__(self, in_channels, growth, dropout):
		super(DensLayer, self).__init__()

		layers = []

		layers.extend([
			nn.Conv2d(in_channels, in_channels, kernel_size=1,
				stride=1, padding=0, bias=True),
			nn.ReLU(inplace=True),])

		if dropout>0:
			layers.append(nn.Dropout(p=dropout))

		layers.extend([
			nn.Conv2d(in_channels,growth,kernel_size=3,
				stride=1,padding=1,bias=True),
			nn.ReLU(inplace=True)
			])

		self.nn = nn.Sequential(*layers)

	def forward(self,x):
		h = self.nn(x)
		h = torch.cat([x,h],dim=1)

		return h

x = torch.randn([64,3,32,32])
layer = DensLayer(in_channels=3,growth=4,dropout=0)
layer(x)

class GatedConv2d(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, padding):
		super(GatedConv2d, self).__init__()
		self.in_channels = in_channels
		self.conv = nn.Conv2d(in_channels, out_channels*3,
			kernel_size=kernel_size,padding=padding)

	def forward(self,x):
		h = self.conv(x)
		a,b,c = torch.chunk(h,chunks=3,dim=1)

		return a + b*torch.sigmoid(c)


class DenseBlock(nn.Sequential):

	def __init__(self, in_channels, out_channels, depth, growth,
		dropout=0., gated_conv=False, zero_init=False):
		layers = [DensLayer(in_channels + i*growth,growth,dropout) for i in range(depth)]

		if gated_conv:
			layers.append(GatedConv2d(in_channels + depth*growth, out_channels, kernel_size=1, padding=0))
		else:
			layers.append(nn.Conv2d(in_channels+depth*growth, out_channels,kernel_size=1,padding=0))

		if zero_init:
			nn.init.zeros_(layers[-1].weight)
			if hasattr(layers[-1],'bias'):
				nn.init.zeros_(layers[-1].bias)

		super(DenseBlock,self).__init__(*layers)


class ResidualDenseBlock(nn.Module):
	def __init__(self, in_channels, out_channels, depth, growth,
		dropout=0., gated_conv=False, zero_init=False):
		super(ResidualDenseBlock,self).__init__()

		self.dense = DenseBlock(in_channels = in_channels,
			out_channels = out_channels,
			depth = depth,
			growth = growth,
			dropout = dropout,
			gated_conv = gated_conv,
			zero_init = zero_init)

	def forward(self, x):
		return x + self.dense(x)	

x = torch.randn([64,3,32,32])
layer = ResidualDenseBlock(in_channels=3,out_channels=3,depth=10,growth=4,dropout=0,gated_conv=False, zero_init=False)
layer(x)


class DenseNet(nn.Sequential):
	def __init__(self, in_channels, out_channels, num_blocks,
		mid_channels, depth, growth, dropout,
		gated_conv=False, zero_init=False):

		layers = [nn.Conv2d(in_channels,mid_channels, kernel_size=1, padding=0)]+[ResidualDenseBlock(in_channels=mid_channels,
			out_channels=mid_channels,
			depth=depth,
			growth=growth,
			dropout=dropout,
			gated_conv=gated_conv,
			zero_init=False) for _ in range(num_blocks)] + [nn.Conv2d(mid_channels,out_channels,kernel_size=1,padding=0)]
		if zero_init:
			nn.init.zeros_(layers[-1].weight)
			if hasattr(layers[-1],'bias'):
				nn.init.zeros_(layers[-1].bias)

		super(DenseNet,self).__init__(*layers)






### cifar10 part

data = MyCIFAR10()
train_loader, test_loader = data.get_data_loaders(32)
imgTensor = next(iter(train_loader))



def one_shot_vis(dataloader,nrow):
	assert isinstance(dataloader,DataLoader)
	imgTensor = next(iter(dataloader))

	grid = vutils.make_grid(imgTensor[:nrow*nrow], padding = 4, nrow=nrow)
	grid = grid.permute(1, 2, 0)
	plt.imshow(grid);plt.show()

	return grid

one_shot_vis(train_loader,nrow=4)



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



model = Flow(base_dist=StandardNormal((24,8,8)),
			transforms=[
				UniformDequantization(num_bits=8),
				Augment(StandardUniform((3,32,32)), x_size=3),
				AffineCouplingBijection(net(6)), ActNormBijection2d(6), Conv1x1(6),
				AffineCouplingBijection(net(6)), ActNormBijection2d(6), Conv1x1(6),
				AffineCouplingBijection(net(6)), ActNormBijection2d(6), Conv1x1(6),
				AffineCouplingBijection(net(6)), ActNormBijection2d(6), Conv1x1(6),
				Squeeze2d(), Slice(StandardNormal((12,16,16)), num_keep=12),
				AffineCouplingBijection(net(12)), ActNormBijection2d(12), Conv1x1(12),
				AffineCouplingBijection(net(12)), ActNormBijection2d(12), Conv1x1(12),
				AffineCouplingBijection(net(12)), ActNormBijection2d(12), Conv1x1(12),
				AffineCouplingBijection(net(12)), ActNormBijection2d(12), Conv1x1(12),
				Squeeze2d(), Slice(StandardNormal((24,8,8)), num_keep=24),
				AffineCouplingBijection(net(24)), ActNormBijection2d(24), Conv1x1(24),
				AffineCouplingBijection(net(24)), ActNormBijection2d(24), Conv1x1(24),
				AffineCouplingBijection(net(24)), ActNormBijection2d(24), Conv1x1(24),
				AffineCouplingBijection(net(24)), ActNormBijection2d(24), Conv1x1(24),
				]).to(device)


x

transform = UniformDequantization(num_bits=8)
z,ldj = transform(x)

transform = Augment(StandardUniform((3,32,32)), x_size=3)
z,ldj = transform(z)

transform = AffineCouplingBijection(net(6))
z,ldj = transform(z)

transform = ActNormBijection2d(6)
z,ldj = transform(z)

transform = Conv1x1(6)
z,ldj = transform(z)

transform = Squeeze2d()
z,ldj = transform(z)

transform = Slice(StandardNormal((12,16,16)), num_keep=12)
z,ldj = transform(z)





import math

optimizer = Adam(model.parameters(), lr=1e-4)


print('Training...')
for epoch in range(50):
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
samples = model.sample(64)

vutils.save_image(img.cpu().float()/255, fp='cifar10_data.png', nrow=8)
vutils.save_image(samples.cpu().float()/255, fp='cifar10_aug_flow.png', nrow=8)




# point cloud experiment


from torch.utils.data import DataLoader






	@property
	def num_splits(self):
		return len(self.splits)

	def get_data_loader(self,split, batch_size, shuffle=True, pin_memory=True, num_workers=4):
		return DataLoader(getattr(self,split),batch_size=batch_size, shuffle=shuffle,pin_memory=pin_memory,num_workers=num_workers)

	def get_data_loaders(self, batch_size, shuffle=True, pin_memory=True, num_workers=4):
		data_loaders = [self.get_data_loader(split=split,batch_size=batch_size,shuffle=shuffle,
			pin_memory=pin_memory,num_workers=num_workers) for split in self.splits]
		return data_loaders


import numpy as np
import os
import torch

from collections.abc import Iterable
from skimage import color, io, transform
from sklearn.datasets import make_moons
from torch.utils.data import Dataset


class PlaneDataset(Dataset):
	def __init__(self, num_points, flip_axes=False):
		self.num_points = num_points
		self.flip_axes = flip_axes
		self.data = None
		self.reset()

	def __getitem__(self, item):
		return self.data[item]

	def __len__(self):
		return self.num_points

	def reset(self):
		self._create_data()
		if self.flip_axes:
			x1 = self.data[:,0]
			x2 = self.data[:,1]
			self.data = torch.stack([x2,x1]).t()

	def _create_data(self):
		raise NotImplementedError


class GaussianDataset(PlaneDataset):

	def _create_data(self):
		x1 = torch.randn(self.num_points)
		x2 = 0.5*torch.randn(self.num_points)
		self.data = torch.stack((x1,x2)).t()


class CrescentDataset(PlaneDataset):

	def _create_data(self):
		x1 = torch.randn(self.num_points)
		x2_mean = 0.5*x1**2 -1
		x2_var = torch.exp(torch.Tensor([-2]))
		x2 = x2_mean + x2_var ** 0.5*torch.randn(self.num_points)
		self.data = torch.stack((x2,x1)).t()

class CrescentCubedDataset(PlaneDataset):
	def _create_data(self):
		x1 = torch.randn(self.num_points)
		x2_mean = 0.2*x1**3
		x2_var = torch.ones(x1.shape)
		x2 = x2_mean + x2_var ** 0.5 * torch.randn(self.num_points)
		self.data = torch.stack((x1,x2)).t()

class SineWaveDataset(PlaneDataset):
	def _create_data(self):
		x1 = torch.randn(self.num_points)
		x2_mean = torch.sin(5*x1)
		x2_var = torch.exp(-2*torch.ones(x1.shape))
		x2 = x2_mean + x2_var ** 0.5*torch.randn(self.num_points)
		self.data = torch.stack((x1,x2)).t()

class AbsDataset(PlaneDataset):
	def _create_data(self):
		x1 = torch.randn(self.num_points)
		x2_mean = torch.abs(x1)-1
		x2_var = torch.exp(-3*torch.ones(x1.shape))
		x2 = x2_mean + x2_var ** 0.5 * torch.randn(self.num_points)
		self.data = torch.stack((x1,x2)).t()

class SignDataset(PlaneDataset):
	def _create_data(self):
		x1 = torch.randn(self.num_points)
		x2_mean = torch.sign(x1) + x1
		x2_var = torch.exp(-3*torch.ones(x1.shape))
		x2 = x2_mean + x2_var**0.5*torch.randn(self.num_points)
		self.data = torch.stack((x1,x2)).t()

class FourCirclesDataset(PlaneDataset):
	def __init__(self,num_points, flip_axes=False):
		if num_points % 4 !=0:
			raise ValueError('Number of data points must be a multiple of four')
		super().__init__(num_points,flip_axes)

	@staticmethod
	def create_circle(num_per_circle, std=0.1):
		u = torch.rand(num_per_circle)
		x1 = torch.cos(2*np.pi*u)
		x2 = torch.sin(2*np.pi*u)
		data = 2*torch.stack((x1,x2)).t()
		data += std * torch.randn(data.shape)
		return data

	def _create_data(self):
		num_per_circle = self.num_points //4
		centers = [
		[-1,-1],
		[-1,1],
		[1,-1],
		[1,1]]

		self.data = torch.cat([self.create_circle(num_per_circle)-torch.Tensor(center) for center in centers])

class DiamondDataset(PlaneDataset):
	def __init__(self,num_points, flip_axes=False, width=20, bound=2.5, std = 0.04):

		self.width = width
		self.bound = bound
		self.std = std
		super().__init__(num_points, flip_axes)

	def _create_data(self, rotate=True):
		means = np.array([
			(x+1e-3 * np.random.rand(),y+1e-3 * np.random.rand())
			for x in np.linspace(-self.bound, self.bound, self.width)
			for y in np.linspace(-self.bound, self.bound, self.width)
			])

		covariance_factor = self.std * np.eye(2)

		index = np.random.choice(range(self.width**2),size=self.num_points, replace=True)
		noise = np.random.randn(self.num_points,2)
		self.data = means[index] + noise@covariance_factor
		if rotate:
			rotation_matrix = np.array([
				[1/np.sqrt(2), -1/np.sqrt(2)],
				[1/np.sqrt(2), 1/np.sqrt(2)]])
			self.data = self.data @ rotation_matrix

		self.data = self.data.astype(np.float32)
		self.data = torch.Tensor(self.data)

class TwoSpiralsDataset(PlaneDataset):
	def _create_data(self):
		n = torch.sqrt(torch.rand(self.num_points//2))* 540*(2*np.pi)/360
		d1x = -torch.cos(n)*n + torch.rand(self.num_points//2)*0.5
		d1y = torch.sin(n)*n + torch.rand(self.num_points//2)*0.5
		x = torch.cat([torch.stack([d1x,d1y]).t(),torch.stack([-d1x,-d1y]).t()])
		self.data = x/3 + torch.randn_like(x)*0.1

class TestGridDataset(PlaneDataset):
	def __init__(self, num_points_per_axis, bounds):
		self.num_points_per_axis = num_points_per_axis
		self.bounds = bounds
		self.shape = [num_points_per_axis] * 2
		self.X = None
		self.Y = None
		super().__init__(num_points = num_points_per_axis**2)

	def _create_data(self):
		x = np.linspace(self.bounds[0][0], self.bounds[0][1], self.num_points_per_axis)
		y = np.linspace(self.bounds[1][0], self.bounds[1][1], self.num_points_per_axis)
		self.X, self.Y = np.meshgrid(x,y)
		data_ = np.vstack([self.X.flatten(),self.Y.flatten()]).T
		self.data = torch.tensor(data_).float()

class CheckerboardDataset(PlaneDataset):
	def _create_data(self):
		x1 = torch.rand(self.num_points)*4-2
		x2_ = torch.rand(self.num_points) - torch.randint(0,2,[self.num_points]).float()*2
		x2 = x2_ + torch.floor(x1)%2
		self.data = torch.stack([x1,x2]).t() * 2

class TwoMoonsDataset(PlaneDataset):

	def _create_data(self):
		data = make_moons(n_samples=self.num_points, noise=0.1, random_state=0)[0]
		data = data.astype('float32')
		data = data*2 + np.array([-1, -0.2])
		self.data = torch.from_numpy(data).float()

class FaceDataset(PlaneDataset):

	def __init__(self, num_points, name='einstein', resize=[512,512], flip_axes = False):
		self.name = name
		self.image = None
		self.resize = resize if isinstance(resize, Iterable) else [resize, resize]
		super().__init__(num_points, flip_axes)

	def _create_data(self):
		root = './'
		path = os.path.join(root,'faces', self.name + '.jpg')
		try:
			image = io.imread(path)
		except FileNotFoundError:
			raise RuntimeError('Unknown face name: {}'.format(self.name))
		image = color.rgb2gray(image)
		self.image = transform.resize(image,self.resize)

		grid = np.array([
			(x,y) for x in range(self.image.shape[0]) for y in range(self.image.shape[1])])

		rotation_matrix = np.array([
			[0,-1],
			[1,0]])

		p = self.image.reshape(-1)/sum(self.image.reshape(-1))
		ix = np.random.choice(range(len(grid)), size=self.num_points, replace=True,p=p)
		points = grid[ix].astype(np.float32)
		points += np.random.rand(self.num_points, 2)
		points /= (self.image.shape[0])

		self.data = torch.tensor(points @ rotation_matrix).float()
		self.data[:,1] += 1

class Gaussian():

	def __init__(self,train_samples=100,test_samples=100):

		self.train = GaussianDataset(num_points=train_samples)
		self.test = GaussianDataset(num_points=test_samples)
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

class Crescent():

	def __init__(self,train_samples=100,test_samples=100):

		self.train = CrescentDataset(num_points=train_samples)
		self.test = CrescentDataset(num_points=test_samples)
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

class CrescentCubed():

	def __init__(self,train_samples=100,test_samples=100):

		self.train = CrescentCubedDataset(num_points=train_samples)
		self.test = CrescentCubedDataset(num_points=test_samples)
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

class SineWave():

	def __init__(self,train_samples=100,test_samples=100):

		self.train = SineWaveDataset(num_points=train_samples)
		self.test = SineWaveDataset(num_points=test_samples)
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

class Abs():

	def __init__(self,train_samples=100,test_samples=100):

		self.train = AbsDataset(num_points=train_samples)
		self.test = AbsDataset(num_points=test_samples)
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

class Sign():

	def __init__(self,train_samples=100,test_samples=100):

		self.train = SignDataset(num_points=train_samples)
		self.test = SignDataset(num_points=test_samples)
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

class FourCircles():

	def __init__(self,train_samples=100,test_samples=100):

		self.train = FourCirclesDataset(num_points=train_samples)
		self.test = FourCirclesDataset(num_points=test_samples)
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

class Diamond():

	def __init__(self,train_samples=100,test_samples=100):

		self.train = DiamondDataset(num_points=train_samples)
		self.test = DiamondDataset(num_points=test_samples)
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

class TwoSpirals():

	def __init__(self,train_samples=100,test_samples=100):

		self.train = TwoSpiralsDataset(num_points=train_samples)
		self.test = TwoSpiralsDataset(num_points=test_samples)
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

class TwoMoons():

	def __init__(self,train_samples=100,test_samples=100):

		self.train = TwoMoonsDataset(num_points=train_samples)
		self.test = TwoMoonsDataset(num_points=test_samples)
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

class Checkerboard():

	def __init__(self,train_samples=100,test_samples=100):

		self.train = CheckerboardDataset(num_points=train_samples)
		self.test = CheckerboardDataset(num_points=test_samples)
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

class Face():

	def __init__(self,train_samples=100,test_samples=100):

		self.train = FaceDataset(num_points=train_samples)
		self.test = FaceDataset(num_points=test_samples)
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





DataSets = Crescent(train_samples=5000, test_samples=5000)
train_loader, test_loader = DataSets.get_data_loaders(128)


imgTensor = next(iter(train_loader))
x = imgTensor


plt.scatter(DataSets.train.data[:,0],DataSets.train.data[:,1]);plt.show()


### point cloud experiment done









import numpy as np
import os 
import pickle

from torch.utils import data

class SpatialMNISTDataset(data.Dataset):

	def __init__(self, data_dir = './spatial_mnist', split='train'):
		
		splits = {
		'train':slice(0,50000),
		'valid':slice(50000,60000),
		'test':slice(60000,70000)
		}

		spatial_path = os.path.join(data_dir, 'spatial.pkl')
		with open(spatial_path,'rb') as file:
			spatial = pickle.load(file)

		labels_path = os.path.join(data_dir, 'labels.pkl')
		with open(labels_path, 'rb') as file:
			labels = pickle.load(file)

		self._spatial = np.array(spatial[splits[split]]).astype(np.float32)
		self._labels = np.array(labels[splits[split]])

		assert len(self._spatial) == len(self._labels)
		self._n = len(self._spatial)

	def __getitem__(self, item):
		return self._spatial[item]

	def __len__(self):
		return self._n



class DataContainer():
	def __init__(self, train, valid, test):
		self.train = train
		self.valid = valid 
		self.test = test




import os
from torch.utils.data import DataLoader
from datasets import SpatialMNISTDataset

DATA_PATH = './'

dataset_choices = {'spatial_mnist'}

def get_data(args):
	assert args.dataset in dataset_choices


	if args.dataset == 'spatial_mnist':
		dataset = DataContainer(SpatialMNISTDataset(os.path.join('./','spatial_mnist'),split='train'),
			SpatialMNISTDataset(os.path.join('./','spatial_mnist'),split='valid'),
			SpatialMNISTDataset(os.path.join('./','spatial_mnist'),split='test'))

	train_loader = DataLoader(dataset.train, batch_size=args.batch_size, shuffle=True)
	valid_loader = DataLoader(dataset.valid, batch_size=args.batch_size, shuffle=False)
	test_loader = DataLoader(dataset.test, batch_size=args.batch_size, shuffle=False)

	return train_loader, valid_loader, test_loader


from prettytable import PrettyTable


def get_args_table(args_dict):
	table = PrettyTable(['Arg','Value'])
	for arg, val in args_dict.item():
		table.add_row([arg,val])

	return table





import os 
import torch
import argparse

import matplotlib.pyplot as plt
from utils import get_args_table
from torch.utils.tensorboard import SummaryWriter


from data import get_data, dataset_choices

import torch.nn as nn




class ScalarAffineBijection(Bijection):

	def __init__(self, shift=None, scale=None):
		super(ScalarAffineBijection, self).__init__()
		assert isinstance(shift, float) or shift is None, 'shift must be a float or None'
		assert isinstance(scale, float) or scale is NOne, 'scale must be a float or None'

		if shift is None and scale is None:
			raise ValueError('At Least one of scale and shift must be provided.')
		if scale == 0:
			raise ValueError('Scale can not be zero.')

		self.register_buffer('_shift',torch.tensor(shift if (shift is not None) else 0.))
		self.register_buffer('_scale',torch.tensor(scale if (scale is not None) else 1.))

	@property 
	def _log_scale(self):
		return torch.log(torch.abs(self._scale))

	def forward(self, x):
		batch_size = x.shape[0]
		num_dims = x.shape[1:].numel()
		z = x*self._scale + self._shift
		ldj = torch.full([batch_size], self._log_scale*num_dims, device=x.device, dtype=x.dtype)

		return z, ldj

	def inverse(self,z):
		batch_size = z.shape[0]
		num_dims = z.shape[1:].numel()
		x = (z - self._shift)/self._scale

		return x



class Permute(Bijection):

	def __init__(self, permutation, dim=1):
		super(Permute, self).__init__()
		assert isinstance(dim, int), 'dim must be an integer'
		assert dim >= 1, 'dim must be >= 1 (0 corresponding to batch dimension)'
		assert isinstance(permutation, torch.Tensor) or isinstance(permutation, Iterable), 'permutation must be a torch.Tensor or Iterable'
		if isinstance(permutation, torch.Tensor):
			assert permutation.ndimension() == 1, 'permutation must be an 1D tensor, but was of shape {}'.format(permutation.shape)
		else:
			permutation = torch.tensor(permutation)

		self.dim = dim
		self.register_buffer('permutation',permutation)


	@propert 
	def inverse_permutation(self):
		return torch.argsort(self.permutation)
	def forward(self,x):
		return torch.index_select(x, self.dim, self.permutation), torch.zeros(x.shape[0],device=x.device, dtype=x.dtype)

	def inverse(self,z):
		return torch.index_select(z, self.dim, self.inverse_permutation)


class Shuffle(Permute):

	def __init__(self, dim_size, dim=1):
		super(Shuffle, self).__init__(torch.randperm(dim_size),dim)

class Reverse(Permute):

	def __init__(self, dim_size, dim=1):
		super(Reverse, self).__init__(torch.arange(dim_size-1, -1,-1),dim)


class PermuteAxes(Bijection):

	def __init__(self, permutation):
		super(PermuteAxes, self).__init__()
		assert isinstance(permutation, Iterable), 'permutation must be an Iterable'
		assert permutation[0] == 0, 'First element of permutation must be 0 (such that batch dimension stays intact)'

		self.permutation = permutation
		self.inverse_permutation = torch.argsort(torch.tensor(self.permutation)).tolist()

	def forward(self, x):
		z = x.permute(self.permutation).contiguous()
		ldj = torch.zeros((x.shape[0],),device=x.device, dtype=x.dtype)
		return z,ldj

	def inverse(self,z):
		x = z.permute(self.inverse_permutation).contiguous()
		return x

class StochasticPermutation(StochasticPermutation):

	def __init__(self, dim=1):
		super(StochasticPermutation, self).__init__()
		self.register_buffer('buffer',torch.zeros(1))
		self.dim = dim

	def forward(self,x):
		rand = torch.rand(x.shape[0], x.shape[self.dim], device=x.device)
		permutation = rand.argsort(dim=1)

		for d in range(1, self.dim):
			permutation = permutation.unsqueeze(1)

		for d in range(self.dim +1, x.dim()):
			permutation = permutation.unsqueeze(-1)

		permutation = permutation.expand_as(x)
		z = torch.gather(x, self.dim, permutation)
		ldj = self.buffer.new_zeros(x.shape[0])
		return z,ldj

	def inverse(self,z):
		rand = torch.rand(z.shape[0], z.shape[self.dim], device=z.device)
		permutation = rand.argsort(dim=1)
		for d in range(1, self.dim):
			permutation = permutation.unsqueeze(1)
		for d in range(self.dim+1, z.dim()):
			permutation = permutation.unsqueeze(-1)
		permutation = permutation.expand_as(z)
		x = torch.gather(z, self.dim, permutation)

		return x



class Reshape(Bijection):

	def __init__(self, input_shape, output_shape):
		super(Reshape, self).__init__()
		self.input_shape = torch.Size(input_shape)
		self.output_shape = torch.Size(output_shape)
		assert self.input_shape.numel() == self.output_shape.numel()

	def forward(self,x):
		batch_size = (x.shape[0],)
		z = x.reshape(batch_size,+ self.output_shape)
		ldj = torch.zeros(batch_size, device=x.device, dtype=x.dtype)

		return z,ldj

	def inverse(self,z):
		batch_size = (z.shape[0],)
		x = z.reshape(batch_size + self.input_shape)
		return x


class Unsqueeze2d(Squeeze2d):

	def __init__(self, factor=2, ordered=False):
		super(Unsqueeze2d, self).__init__(factor=factor, ordered=ordered)

	def forward(self,x):
		z = self._unsqueeze(x)
		ldj = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
		return z, ldj 

	def inverse(self,z):
		x = self._squeeze(z)
		return x


class Rotate(Bijection):

	def __init__(self, degrees, dim1, dim2):

		super(Rotate, self).__init__()
		assert isinstance(degrees, int), 'degrees must be an integer'
		assert isinstance(dim1, int), 'dim1 must be an integer'
		assert isinstance(dim2, int), 'dim2 must be an integer'

		assert degrees in {90,180,270}
		assert dim1 !=0
		assert dim2 != 0
		assert dim1 != dim2

		self.degrees = degrees
		self.dim1 = dim1
		self.dim2 = dim2

	def _rotate90(self,x):

		return x.transpose(self.dim1, self.dim2).flio(self.dim1)

	def _rotate90_inv(self,z):
		return z.flip(self.dim1).transpose(self.dim1,self.dim2)

	def _rotate180(self,x):
		return x.flip(self.dim1).flip(self.dim2)

	def _rotate180_inv(self,z):
		return z.flip(self.dim2).flip(self.dim1)

	def _rotate270(self,x):
		return x.transpose(self.dim1, self.dim2).flip(self.dim2)

	def _rotate270_inv(self,z):
		return z.flip(self.dim2).transpose(self.dim1,self.dim2)

	def forward(self,x):

		if self.degrees == 90: z = self._rotate90(x)
		elif self.degrees == 180: z = self._rotate180(x)
		elif self.degrees == 270: z = self._rotate270(x)

		ldj = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

		return x, torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

	def inverse(self,z):
		if self.degrees == 90: x = self._rotate90_inv(z)
		elif self.degrees == 180: x = self._rotate180_inv(z)
		elif self.degrees == 270: x = self._rotate270_inv(z)
		return z

import torch
import torch.nn.functional as F


def scale_fn(scale_str):
	assert scale_str in {'exp','softplus', 'sigmoid', 'tanh_exp'}
	if scale_str == 'exp': 	return lambda s: torch.exp(s)
	elif scale_str == 'softplus':	return lambda s: F.softplus(s)
	elif scale_str == 'sigmoid': return lambda s: torch.sigmoid(s+2.) + 1e-3
	elif scale_str == 'tanh_exp': return lambda s: torch.exp(2.*torch.tanh(s/2.))



class MultiscaleDenseNet(nn.Module):
	def __init__(self, in_channels, out_channels, num_scales, num_blocks, mid_channels,
		depth, growth, dropout, gated_conv=False, zero_init=False):

		super(MultiscaleDenseNet, self).__init__()
		assert num_scales >1
		self.num_scales = num_scales


		def get_densenet(cin, cout, zinit=False):
			return DenseNet(in_channels=cin,
				out_channels=cout,
				num_blocks=num_blocks,
				mid_channels=mid_channels,
				depth=depth,
				growth=growth,
				dropout=dropout,
				gated_conv=gated_conv,
				zero_init=zinit)

		self.down_in = get_densenet(in_channels, mid_channels)

		down = []
		for i in range(num_scales -1):
			down.append(nn.Sequential(nn.Conv2d(mid_channels, mid_channels, kernel_size=2, padding=0,stride=2),
				get_densenet(mid_channels,mid_channels)))

		self.down = nn.ModuleList(down)

		up = []

		for i in range(num_scales -1):
			np.append(nn.Sequential(get_densenet(mid_channels,mid_channels),
				nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size=2, padding=1, stride=2)))

		self.up = nn.ModuleList(up)

		self.up_out = get_densenet(mid_channels, out_channels, zinit=zero_init)


	def forward(self,x):

		d = [self.down_in(x)]

		for down_layer in self.down:
			d.append(down_layer(d[-1]))

		u = [d[-1]]
		for i, up_layer in enumerate(self.up):
			u.append(up_layer(u[-1])+d[self.num_scales -2-i])

		return self.up_out(u[-1])


import math
import torch
import torch.nn as nn

class PositionalEncodeingImage(nn.Module):

	def __init__(self, image_shape, embedding_dim):
		super(PositionalEncodeingImage, self).__init__()
		assert len(image_shape) == 3, 'image shape should have length 3: (C,H,W)'
		self.image_shape = image_shape
		self.embedding_dim = embedding_dim

		c,h,w = image_shape
		self.encode_c = nn.Parameter(torch.Tensor(1,c,1,1,embedding_dim))
		self.encode_h = nn.Parameter(torch.Tensor(1,1,h,1,embedding_dim))
		self.encode_w = nn.Parameter(torch.Tensor(1,1,1,w,embedding_dim))

		self.reset_parameters()

	def reset_parameters(self):

		nn.init.normal_(self.encode_c, std=0.125/math.sqrt(3*self.embedding_dim))
		nn.init.normal_(self.encode_h, std=0.125/math.sqrt(3*self.embedding_dim))
		nn.init.normal_(self.encode_w, std=0.125/math.sqrt(3*self.embedding_dim))

	def forward(self,x):
		return x + self.encode_c + self.encode_h + self.encode_w


layer = PositionalEncodeingImage(image_shape=(3,32,32),embedding_dim=10)
layer(x)




class AutoregressiveShift(nn.Module):

	def __init__(self, embed_dim):
		super(AutoregressiveShift, self).__init__()
		self.embed_dim = embed_dim
		self.first_token = nn.Parameter(torch.Tensor(1,1,embed_dim))
		self._reset_parameters()

	def _reset_paramters(self):
		nn.init.xavier_uniform_(self.first_token)

	def forward(self,x):
		first_token = self.first_token.expand(1,x.shape[1],self.embed_dim)
		return torch.cat([first_token, x[:-1]], dim=0)


layer = PositionalEncodeingImage(image_shape=(3,32,32),embedding_dim=10)
layer(x)



def _prep_zigzag_cs(channels, height, width):

	diagonals = [[] for i in range(height+width-1)]

	for i in range(height):
		for j in range(width):
			sum = i+j
			if(sum%2==0):
				diagonals[sum].insert(0,(i,j))
			else:
				diagonals[sum].append((i,j))

	idx_list = []
	for d in diagonals:
		for idx in d:
			for c in range(channels):
				idx_list.append((c,)+idx)

	idx0,idx1,idx2 = zip(*idx_list)
	return idx0,idx1,idx2


class Image2Seq(nn.Module):

	def __init__(self, autoregressive_order, image_shape):
		assert autoregressive_order in {'cwh','whc','zigzag_cs'}
		super(Image2Seq, self).__init__()
		self.autoregressive_order = autoregressive_order
		self.channels = image_shape[0]
		self.height = image_shape[1]
		self.width = image_shape[2]
		if autoregressive_order == 'zigzag_cs':
			self.idx0, self.idx1, self.idx2 = _prep_zigzag_cs(self.channels, self.height, self.width)

	def forward(self, x):
		b, dim = x.shape[0], x.shape[-1]
		l = x.shape[1:-1].numel()
		if self.autoregressive_order == 'whc':

			x = x.permute([1,2,3,0,4])

			x = x.reshape(l,b,dim)

		elif self.autoregressive_order == 'cwh':

			x = x.permute([2,3,1,0,4])

			x = x.reshape(l,b,dim)

		elif self.autoregressive_order == 'zigzag_cs':

			x = x[:, self.idx0, self.idx1, self.idx2, :]

			x = x.permute([1,0,2])

		return x


class Seq2Image(nn.Module):

	def __init__(self, autoregressive_order, image_shape):
		assert autoregressive_order in {'cwh','whc','zigzag_cs'}
		super(Seq2Image, self).__init__()
		self.autoregressive_order = autoregressive_order
		self.channles = channles
		self.height = height
		self.width = width
		if autoregressive_order == 'zigzag_cs':
			self.idx0, self.idx1, self.idx2 = _prep_zigzag_cs(self.channels, self.height, self.width)

	def forward(self,x):
		b, dim = x.shape[1], x.shape[2]
		if self.autoregressive_order == 'whc':
			x = x.reshape(self.channels, self.height, self.width, b, dim)

			x = x.permute([3,0,1,2,4])

		elif self.autoregressive_order == 'cwh':
			x = x.reshape(self.height, self.width, self.channels, b,dim)
			x = x.permute([3,2,0,1,4])

		elif self.autoregressive_order == 'zigzag_cs':
			x = x.permute([1,0,2])
			y = torch.empty((x.shape[0],self.channels, self.height, self.width, x.shape[-1]),dtype=x.dtype, device=x.device)
			y[:, self.idx0,self.idx1, self.idx2,:] = x

			x = y

		return x



from torch.utils import checkpoint

class DenseTransformerBlock(nn.Module):

	def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1,activation='gelu',kdim=None, vdim=None, attn_bias=True, checkpoint=False):
		super(DenseTransformerBlock, self).__init__()
		self.self_attn = nn.MultiheadAttention(d_model,nhead, dropout=dropout, kdim=kdim, vdim=vdim,bias=attn_bias)

		self.linear1 = nn.Linear(d_model, dim_feedforward)
		self.linear2 = nn.Linear(dim_feedforward, d_model)

		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)
		self.dropout1 = nn.Dropout(dropout)
		self.dropout2 = nn.Dropout(dropout)

		self.activation = act_module(activation)
		self.checkpoint = checkpoint

		self._reset_parameters()


	def _reset_parameters(self):
		nn.init.normal_(self.linear1.weight, std=0.125/math.sqrt(self.linear1.weight.shape[1]))
		nn.init.normal_(self.linear2.weight, std=0.125/math.sqrt(self.linear2.weight.shape[1]))

		nn.init.zeros_(self.linear1.bias)
		nn.init.zeros_(self.linear2.bias)

		nn.init.normal_(self.self_attn.in_proj_weight, std = 0.1245/math.sqrt(self.self_attn.in_proj_weight.shape[1]))
		if not self.self_attn._qkv_same_embed_dim:
			nn.init.normal_(self.self_attn.q_proj_weight,std=0.125/math.sqrt(self.self_attn.q_proj_weight.shape[1]))
			nn.init.normal_(self.self_attn.k_proj_weight, std=0.125/math.sqrt(self.self_attn.k_proj_weight.shape[1]))
			nn.init.normal_(self.self_attn.v_proj_weight, std=0.125/math.sqrt(self.self_attn.v_proj_weight.shape[1]))

		if self.self_attn.in_proj_bias is not None:
			nn.init.zeros_(self.self_attn.in_proj_bias)

		nn.init.normal_(self.self_attn.out_proj.weight, std=0.125/math.sqrt(self.self_attn.out_proj.weight.shape[1]))

		if self.self_attn.out_proj.bias is not None:
			nn.init.zeros_(self.self_attn.out_proj.bias)


	def _attn_block(self, x, attn_mask=None, key_padding_mask=None):
		x = self.norm1(x)
		x = self.self_attn(x,x,x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)[0]
		x = self.dropout1(x)
		return x

	def _ff_block(self,x):
		x = self.norm2(x)
		x = self.linear2(self.activation(self.linear1(x)))
		x = self.dropout2(x)
		return x

	def _forward(self, x, attn_mask=None, key_padding_mask=None):
		ax = self._attn_block(x, attn_mask=attn_mask,key_padding_mask=key_padding_mask)
		bx = self._ff_block(x+ax)
		return x + ax+bx

	def forward(self, x, attn_mask=None, key_padding_mask=None):
		if not self.checkpoint:
			return self._forward(x,attn_mask, key_padding_mask)
		else:
			x.requires_grad_(True)
			return checkpoint.checkpoint(self._forward, x, attn_mask, key_padding_mask)


def _get_clones(module, N):
	return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

layer = DenseTransformerBlock(d_model=512,nhead=4, 
	dim_feedforward=512, dropout=0.1,activation='gelu', attn_bias=True, checkpoint=False)

self_attn = nn.MultiheadAttention(512,4, dropout=0.1,bias=True)

class DenseTransformer(nn.Module):

	def __init__(self, d_model=512, nhead=8,
		num_layers=6, dim_feedforward=512, dropout=0.1,
		activation='gelu',kdim=None, vdim=None,
		attn_bias=True, checkpoint_blocks=False):

		super(DenseTransformer, self).__init__()

		decoder_layer = DenseTransformerBlock(d_model=d_model,nhead=nhead,dim_feedforward=dim_feedforward,
			dropout=dropout,activation=activation,kdim=kdim,vdim=vdim,attn_bias=attn_bias,checkpoint=checkpoint_blocks)

		self.layers = _get_clones(decoder_layer, num_layers)
		self.out_norm = nn.LayerNorm(d_model)

		self.num_layers = num_layers
		self.d_model = d_model
		self.nhead = nhead

		self._reset_parameters()


	def forward(self, x, key_padding_mask=None):
		if x.size(2) != self.d_model:
			raise RuntimeError('the feature number of src and tgt must be equal to d_model')

		attn_mask = self.generate_square_subsequent_mask(x.shape[0]).to(x.device)

		for decoder_layer in self.layers:
			x = decoder_layer(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

		return self.out_norm(x)

	def generate_square_subsequent_mask(self, sz):

		mask = (torch.triu(torch.ones(sz,sz))==1).transpose(0,1)
		mask = mask.float().masked_fill(mask ==0, float('-inf')).masked_fill(mask ==1, float(0.0))

		return mask

	def _reset_parameters(self):

		for p in self.parameters():
			if p.dim()>1:
				nn.init.xavier_uniform_(p)




import copy

class PositionalEncoding1d_no_embedding(nn.Module):

	def __init__(self, size, embedding_dim):
		super(PositionalEncoding1d, self).__init__()
		self.size = size
		self.embedding_dim = embedding_dim
		self.encode_l = nn.Parameter(torch.Tensor(size,embedding_dim))
		self.reset_parameters()

	def reset_parameters(self):

		nn.init.normal_(self.encode_l, std=0.125/math.sqrt(self.embedding_dim))

	def forward(self,x):
		return x + self.encode_l

class PositionalEncoding1d(nn.Module):

	def __init__(self, size, embedding_dim):
		super(PositionalEncoding1d, self).__init__()
		self.size = size
		self.embedding_dim = embedding_dim
		self.encode_l = nn.Parameter(torch.Tensor(size,1,embedding_dim))
		self.reset_parameters()

	def reset_parameters(self):

		nn.init.normal_(self.encode_l, std=0.125/math.sqrt(self.embedding_dim))

	def forward(self,x):
		return x + self.encode_l


layer = PositionalEncoding1d(size=50,embedding_dim=2)
layer(x)

class PositionalDenseTransformer_no_embedding(nn.Module):
	def __init__(self, l_input=50, d_input=2, d_output=2, d_model=512, nhead=8,
		num_layers=6, dim_feedforward=512, dropout=0.1,
		activation='gelu',kdim=None, vdim=None,
		attn_bias=True, checkpoint_blocks=False,
		in_lambda= lambda x:x,
		out_lambda = lambda x:x):

		super(PositionalDenseTransformer,self).__init__()

		decoder_layer = DenseTransformerBlock(d_model=d_model,
										nhead=nhead,
										dim_feedforward=dim_feedforward,
										dropout=dropout,
										activation=activation,
										attn_bias=attn_bias,
										checkpoint=checkpoint_blocks)

		self.in_lambda = LambdaLayer(in_lambda)
		self.in_linear = nn.Linear(d_input, d_model)
		self.encode = PositionalEncoding1d_no_embedding(l_input, d_model)
		self.layers = _get_clones(decoder_layer, num_layers)
		self.out_norm = nn.LayerNorm(d_model)
		self.out_linear = nn.Linear(d_model, d_output)
		self.out_lambda = LambdaLayer(out_lambda)

		self.num_layers = num_layers
		self.d_model = d_model
		self.nhead = nhead

		self._reset_parameters()

	def forward(self,x):

		x = self.in_lambda(x)

		x = self.in_linear(x)
		x = self.encode(x)

		for decoder_layer in self.layers:
			x = decoder_layer(x, attn_mask=None, key_padding_mask=None)

		x = self.out_norm(x)
		x = self.out_linear(x)

		x = self.out_lambda(x)

		return x

	def _reset_parameters(self):

		for decoder_layer in self.layers:
			decoder_layer.linear2.weight.data /= math.sqrt(2*self.num_layers)
			decoder_layer.self_attn.out_proj.weight.data /= math.sqrt(2*self.num_layers)


		nn.init.zeros_(self.out_linear.weight)
		if self.out_linear.bias is not None:
			nn.init.zeros_(self.out_linear.bias)




layer = PositionalDenseTransformer(l_input=50, d_input=2, d_output=2, d_model=512, nhead=8,
		num_layers=6, dim_feedforward=512, dropout=0.1,
		activation='relu',kdim=None, vdim=None,
		attn_bias=True, checkpoint_blocks=False,
		in_lambda= lambda x:x,
		out_lambda = lambda x:x)

layer(x)



model = PositionalDenseTransformer()



class Args(object):
	def __init__(self):
		self.dataset = None
		self.batch_size = None


args = Args()
args.dataset = 'spatial_mnist'
args.batch_size = 64

train_loader, valid_loader, test_loader = get_data(args)

x = next(iter(test_loader))[:64]



encode = PositionalEncoding1d(50, 2)


encode(x).shape








import warnings 
import copy


def repeat_rows(x, num_reps):

	shape = x.shape
	x = x.unsqueeze(1)
	x = x.expand(shape[0],num_reps, *shape[1:])
	return merge_leading_dims(x, num_dims=2)


class MaskedLinear(nn.Linear):

	def __init__(self,
		in_degrees,
		out_features,
		data_features,
		random_mask=False,
		random_seed=None,
		is_output=False,
		data_degrees=None,
		bias=True):

		if is_output:
			assert data_degrees is not None
			assert len(data_degrees) == data_features

		super(MaskedLinear, self).__init__(in_features=len(in_degrees),
			out_features=out_features,
			bias=bias)

		self.out_features = out_features
		self.data_features = data_features
		self.is_output = is_output

		mask, out_degrees = self.get_mask_and_degrees(in_degrees=in_degrees,
			data_degrees=data_degrees,
			random_mask=random_mask,
			random_seed=random_seed)

		self.register_buffer('mask',mask)
		self.register_buffer('degrees',out_degrees)

	@staticmethod
	def get_data_degrees(in_features, random_order=False, random_seed=None):
		if random_order:
			rng = np.random.RandomState(random_seed)
			return torch.from_numpy(rng.permutation(in_features)+1)

		else:
			return torch.arange(1,in_features+1)

	def get_mask_and_degrees(self,in_degrees, data_degrees,random_mask, random_seed):
		if self.is_output:
			out_degrees = repeat_rows(data_degrees, self.out_features//self.data_features)
			mask = (out_degrees[...,None]>in_degrees).float()

		else:
			if random_mask:
				min_in_degree = torch.min(in_degrees).item()
				min_in_degree = min(min_in_degree,self.data_features-1)
				rng = np.random.RandomState(random_seed)
				out_degrees = torch.from_numpy(rng.randint(min_in_degree,
					self.data_features,
					size=[self.out_features]))

			else:
				max_ = max(1,self.data_features-1)
				min_ = min(1,self.data_features-1)
				out_degrees = torch.arange(self.out_features)%max_ + min_

			mask = (out_degrees[...,None] >= in_degrees).float()

		return mask, out_degrees

	def update_mask_and_degrees(self,in_degrees,data_degrees,random_mask,random_seed):

		mask, out_degrees = self.get_mask_and_degrees(in_degrees=in_degrees,
			data_degrees=data_degrees,random_mask=random_mask,random_seed=random_seed)

		self.mask.data.copy_(mask)
		self.degrees.data.copy_(out_degrees)

	def forward(self,x):

		return F.linear(x, self.weight*self.mask, self.bias)


class MADE(nn.Sequential):

	def __init__(self, features, num_params, hidden_features, random_order=False, random_mask=False,
		random_seed=None, activation='relu',dropout_prob=0.0,batch_norm=False):

		layers = []

		data_degrees = MaskedLinear.get_data_degrees(features, random_order=random_order,random_seed=random_seed)
		in_degrees = copy.deepcopy(data_degrees)
		for i,out_features in enumerate(hidden_features):
			layers.append(MaskedLinear(in_degrees=in_degrees,out_features=out_features,
				data_features=features,random_mask=random_mask,random_seed=random_seed+i if random_seed else None,
				is_output=False))


			in_degrees = layers[-1].degrees
			if batch_norm:
				layers.append(nn.BatchNorm1d(out_features))
			layers.append(act_module(activation))
			if dropout_prob >0.0:
				layers.append(nn.Dropout(dropout_prob))


		layers.append(MaskedLinear(in_degrees=in_degrees,
			out_features=features*num_params,data_features=features,random_mask=random_mask,
			random_seed=random_seed,is_output=True,data_degrees=data_degrees))

		layers.append(ElementwiseParams(num_params, mode='sequential'))

		super(MADE, self).__init__(*layers)


class AgnosticMADE(MADE):

	def __init__(self, features, num_params, hidden_features, order_agnostic=True,
		connect_agnostic=True, num_masks=16, activation='relu', dropout_prob=0.0, batch_norm=False):

		self.features = features
		self.order_agnostic = order_agnostic
		self.connect_agnostic = connect_agnostic
		self.num_masks = num_masks
		self.current_mask = 0

		super(AgnosticMADE, self).__init__(features=features,num_params=num_params,
			hidden_features=hidden_features,random_order=order_agnostic,random_mask=connect_agnostic,
			random_seed=self.current_mask,activation=activation,dropout_prob=dropout_prob,
			batch_norm=batch_norm)

	def update_masks(self):
		self.current_mask = (self.current_mask+1)%self.num_masks

		data_degrees = MaskedLinear.get_data_degrees(self.features,random_order=self.order_agnostic,
			random_seed=self.current_mask)

		in_degrees = copy.deepcopy(data_degrees)
		for module in self.modules():
			if isinstance(module, MaskedLinear):
				module.update_mask_and_degrees(in_degrees=in_degrees,data_degrees=data_degrees,
					random_mask=self.connect_agnostic,random_seed=self.current_mask)

				in_degrees = module.degrees

	def forward(self,x):
		if self.num_masks>1: self.update_masks()
		return super(AgnosticMADE,self).forward(x)







def mask_conv2d_spatial(mask_type, height, width):

	mask = torch.ones([1,1,height,width])
	mask[:,:, height//2, width//2+(mask_type == 'B'):] = 0
	mask[:, :, height//2+1:] = 0

	return mask



def mask_channels(mask_type, in_channels, out_channels, data_channels=3):

	in_factor = in_channels // data_channels +1
	out_factor = out_channels // data_channels +1

	base_mask = torch.ones([data_channels,data_channels])
	if mask_type =='A':
		base_mask = base_mask.tril(-1)

	else:
		base_mask = base_mask.tril(0)

	mask_p1 = torch.cat([base_mask]*in_factor, dim=1)
	mask_p2 = torch.cat([mask_p1]*out_factor, dim=0)

	mask = mask_p2[0:out_channels,0:in_channels]
	return mask


def mask_conv2d(mask_type, in_channels, out_channels, height, width, data_channels=3):

	mask = torch.ones([out_channels,in_channels,height,width])
	mask[:,:, height//2, width//2] = mask_channels(mask_type,in_channels,out_channels,data_channels)
	mask[:,:, height//2, width//2 +1:] = 0

	mask[:,:,height//2+1:]=0
	return mask

class _MaskedConv2d(nn.Conv2d):

	def register_mask(self, mask):

		self.register_buffer('mask',mask)

	def forward(self, x):
		self.weight.data *= self.mask 
		return super(_MaskedConv2d,self).forward(x)

class SpatialMaskedConv2d(_MaskedConv2d):

	def __init__(self, *args, mask_type, **kwargs):
		super(SpatialMaskedConv2d,self).__init__(*args, **kwargs)
		assert mask_type in {'A','B'}
		_,_, height, width = self.weight.size()
		mask = mask_conv2d_spatial(mask_type,height,width)
		self.register_mask(mask)


class MaskedConv2d(_MaskedConv2d):

	def __init__(self, *args, mask_type, data_channels=3, **kwargs):
		super(MaskedConv2d,self).__init__(*args, **kwargs)
		assert mask_type in {'A','B'}
		out_channels, in_channels, height, width = self.weight.size()

		mask = mask_conv2d(mask_type, in_channels, out_channels, height, width, data_channels)
		self.register_mask(mask)


class MaskedResidualBlock2d(nn.Module):

	def __init__(self, h, kernel_size=3, data_channels=3):
		super(MaskedResidualBlock2d,self).__init__()

		self.conv1 = MaskedConv2d(2*h,h, kernel_size=1, mask_type='B', data_channels=data_channels)
		self.conv2 = MaskedConv2d(h, h, kernel_size=kernel_size,padding=kernel_size//2,mask_type='B',data_channels=data_channels)
		self.conv3 = MaskedConv2d(h,2*h, kernel_size=1, mask_type='B', data_channels=data_channels)

	def forward(self,x):
		identity = x

		x = self.conv1(F.relu(x))
		x = self.conv2(F.relu(x))
		x = self.conv3(F.relu(x))

		return x + identity


class SpatialMaskedResidualBlock2d(nn.Module):
	def __init__(self, h, kernel_size=3):
		super(SpatialMaskedResidualBlock2d,self).__init__()
		self.conv1 = nn.Conv2d(2*h, h, kernel_size=1)
		self.conv2 = SpatialMaskedConv2d(h,h,kernel_size=kernel_size,padding=kernel_size//2,mask_type='B')
		self.conv3 = nn.Conv2d(h, 2*h, kernel_size=1)

	def forward(self,x):
		identity = x

		x = self.conv1(F.relu(x))
		x = self.conv2(F.relu(x))
		x = self.conv3(F.relu(x))

		return x+identity


class PixelCNN(nn.Sequential):

	def __init__(self, in_channels, num_params, filters=128, num_blocks=15, output_filters=1024, kernel_size=3, kernel_size_in=7, init_transforms=lambda x: 2*x-1):

		layers = [LambdaLayer(init_transforms)]+\
			[MaskedConv2d(in_channels, 2*filters, kernel_size=kernel_size_in,padding=kernel_size_in//2, mask_type='A', data_channels=in_channels)]+\
			[MaskedResidualBlock2d(filters, data_channels=in_channels,kernel_size=kernel_size_in) for _ in range(num_blocks)] +\
			[nn.ReLU(True), MaskedConv2d(2*filters, output_filters, kernel_size=1,mask_type='B',data_channels=in_channels)]+\
			[nn.ReLU(True),MaskedConv2d(output_filters, num_params*in_channels, kernel_size=1, mask_type='B',data_channels=in_channels)]+\
			[ElementwiseParams2d(num_params)]

		super(PixelCNN, self).__init__(*layers)


layer = PixelCNN(in_channels=3, num_params=5, filters=5, num_blocks=2, output_filters=15,kernel_size=3, kernel_size_in=3)






class DecoderOnlyTransformerBlock(nn.Module):

	def __init__(self, d_model, nhead, dim_feedforward=2048,dropout=0.1,activation='relu',
		kdim=None,vdim=None,attn_bias=True, checkpoint=False):
		super(DecoderOnlyTransformerBlock,self).__init__()
		self.self_attn = nn.MultiheadAttention(d_model,nhead, dropout=dropout, kdim=kdim,vdim=vdim,bias=attn_bias)

		self.linear1 = nn.Linear(d_model,dim_feedforward)
		self.dropout = nn.Dropout(dropout)
		self.linear2 = nn.Linear(dim_feedforward,d_model)

		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)
		self.dropout1 = nn.Dropout(dropout)
		self.dropout2 = nn.Dropout(dropout)

		self.activation = act_module(activation)
		self.checkpoint = checkpoint

	def _attn_block(self, x, attn_mask=None, key_padding_mask=None):
		x2 = self.self_attn(x,x,x,attn_mask=attn_mask,key_padding_mask=key_padding_mask)[0]
		x = x + self.dropout1(x2)
		x = self.norm1(x)
		return x

	def _ff_block(self,x,attn_mask=None, key_padding_mask=None):
		x2  = self.linear2(self.dropout(self.activation(self.linear1(x))))
		x = x + self.dropout2(x2)
		x = self.norm2(x)
		return x

	def _forward(self, x,attn_mask=None, key_padding_mask=None):
		x = self._attn_block(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
		x = self._ff_block(x)

		return x

	def forward(self, x, attn_mask=None, key_padding_mask=None):
		if not self.checkpoint:
			return self._forward(x,attn_mask, key_padding_mask)
		else:
			x.requires_grad_(True)
			return checkpoint.checkpoint(self._forward, x, attn_mask, key_padding_mask)

class DecoderOnlyTransformer(nn.Module):
	def __init__(self, d_model=512, nhead=8,
		num_layers=6, dim_feedforward=2048, dropout=0.1, activation='relu', kdim=None,
		vdim=None,attn_bias=True, checkpoint_blocks=False):

		super(DecoderOnlyTransformer,self).__init__()

		decoder_layer = DecoderOnlyTransformerBlock(d_model=d_model,
			nhead=nhead,dim_feedforward=dim_feedforward,dropout=dropout,activation=activation,
			kdim=kdim,vdim=vdim,attn_bias=attn_bias,checkpoint=checkpoint_blocks)

		self.layers = _get_clones(decoder_layer, num_layers)
		self.out_norm = nn.LayerNorm(d_model)

		self._reset_parameters()

		self.d_model = d_model
		self.nhead = nhead


	def forward(self, x, key_padding_mask=None):
		if x.size(2) != self.d_model:
			raise RuntimeError('the feature number of src and tgt must be equal to d_model')

		attn_mask = self.generate_square_subsequent_mask(x.shape[0]).to(x.device)

		for decoder_layer in self.layers:
			x = decoder_layer(x,attn_mask=attn_mask,key_padding_mask=key_padding_mask)

		return self.out_norm(x)

	def generate_square_subsequent_mask(self, sz):
		mask = (torch.triu(torch.ones(sz,sz))==1).transpose(0,1)
		mask = mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask==1,float(0.0))
		return mask 

	def _reset_parameters(self):

		for p in self.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p)


layer = DecoderOnlyTransformer()

layer(x)




class AutoregressiveBijection(Bijection):

	def __init__(self, autoregressive_net, autoregressive_order='ltr'):
		super(AutoregressiveBijection, self).__init__()
		assert isinstance(autoregressive_order,str) or isinstance(autoregressive_order,Iterable)
		assert autoregressive_order in {'ltr'}

		self.autoregressive_net = autoregressive_net
		self.autoregressive_order = autoregressive_order

	def forward(self,x):
		elementwise_params = self.autoregressive_net(x)
		z, ldj = self._elementwise_forward(x, elementwise_params)
		return z,ldj

	def inverse(self,z):
		with torch.no_grad():
			if self.autoregressive_order == 'ltr': return self._inverse_ltr(z)

	def _inverse_ltr(self,z):
		x = torch.zeros_like(z)
		for d in range(x.shape[1]):
			elementwise_params  = self.autoregressive_net(x)
			x[:,d] = self._elementwise_inverse(z[:,d],elementwise_params[:,d])

		return x

	def _output_dim_multiplier(self):
		raise NotImplementError()

	def _elementwise_forward(self, x, elementwise_params):
		raise NotImplementError()

	def _elementwise_inverse(self, z, elementwise_params):
		raise NotImplementError()


class AdditiveAutoregressiveBijection(AutoregressiveBijection):

	def _output_dim_multiplier(self):
		return 1

	def _elementwise_forward(self, x, elementwise_params):
		return x + elementwise_params, torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

	def _elementwise_inverse(self,z, elementwise_params):
		return z - elementwise_params



class AffineAutoregressiveBijection(AutoregressiveBijection):

	def __init__(self, autoregressive_net, autoregressive_order='ltr', scale_fn=lambda s:torch.exp(s)):
		super(AffineAutoregressiveBijection, self).__init__(autoregressive_net=autoregressive_net,autoregressive_order=autoregressive_order)
		assert callable(scale_fn)
		self.scale_fn = scale_fn

	def _output_dim_multiplier(self):
		return 2

	def _elementwise_forward(self,x, elementwise_params):
		assert elementwise_params.shape[-1] == self._output_dim_multiplier()
		unconstrained_scale, shift = self._unconstrained_scale_and_shift(elementwise_params)
		scale = self.scale_fn(unconstrained_scale)
		z = scale*x + shift

		ldj = sum_except_batch(torch.log(scale))

		return z,ldj

	def _elementwise_inverse(self, z, elementwise_params):
		assert elementwise_params.shape[-1] == self._output_dim_multiplier()
		unconstrained_scale,shift = self._unconstrained_scale_and_shift(elementwise_params)
		scale = self.scale_fn(unconstrained_scale)
		x = (z-shift)/scale
		return x

	def _unconstrained_scale_and_shift(self, elementwise_params):

		unconstrained_scale = elementwise_params[...,0]
		shift = elementwise_params[...,1]
		return unconstrained_scale,shift


net = MADE_Old(features=3072, num_params=2, hidden_features=[4], random_order=False, random_mask=False,
		random_seed=None, activation='relu',dropout_prob=0.0,batch_norm=False)

layer = AffineAutoregressiveBijection(net)





class AutoregressiveBijection2d(Bijection):

	def __init__(self, autoregressive_net, autoregressive_order='raster_cwh'):
		super(AutoregressiveBijection2d,self).__init__()
		assert isinstance(autoregressive_order,str) or isinstance(autoregressive_order, Iterable)
		assert autoregressive_order in {'raster_cwh','raster_wh'}
		self.autoregressive_net = autoregressive_net
		self.autoregressive_order = autoregressive_order

	def forward(self,x):
		elementwise_params = self.autoregressive_net(x)
		z,ldj = self._elementwise_forward(x,elementwise_params)
		return z,ldj

	def inverse(self,z):
		with torch.no_grad:
			if self.autoregressive_order == 'raster_cwh': return self._inverse_raster_cwh(z)
			if self.autoregressive_order == 'raster_wh': return self._inverse_raster_wh(z)

	def _inverse_raster_cwh(self,z):
		x = torch.zeros_like(z)
		for h in range(x.shape[2]):
			for w in range(x.shape[3]):
				for c in range(x.shape[1]):
					elementwise_params = self.autoregressive_net(x)
					x[:,c,h,w] = self._elementwise_inverse(z[:,c,h,w], elementwise_params[:,c,h,w])

		return x

	def _inverse_raster_wh(self,z):
		x = torch.zeros_like(z)
		for h in range(x.shape[2]):
			for w in range(x.shape[3]):
				elementwise_params = self.autoregressive_net(x)
				x[:,:,h,w] = self._elementwise_inverse(z[:,:,h,w], elementwise_params[:,:,h,w])
		return x

	def _output_dim_multiplier(self):
		raise NotImplementError()

	def _elementwise_forward(self,x,elementwise_params):
		raise NotImplementError()

	def _elementwise_inverse(self,z,elementwise_params):
		raise NotImplementError()


class AdditiveAutoregressiveBijection2d(AutoregressiveBijection2d):

	def _output_dim_multiplier(self):
		return 1

	def _elementwise_forward(self, x, elementwise_params):
		return x + elementwise_params, torch.zeros(x.shape[0],device=x.device, dtype=x.dtype)

	def _elementwise_inverse(self, z, elementwise_params):
		return z - elementwise_params


class AffineAutoregressiveBijection2d(AutoregressiveBijection2d):

	def __init__(self, autoregressive_net, autoregressive_order='raster_cwh',scale_fn=lambda s:torch.exp(s)):
		super(AffineAutoregressiveBijection2d,self).__init__(autoregressive_net=autoregressive_net,autoregressive_order=autoregressive_order)
		assert callable(scale_fn)
		self.scale_fn = scale_fn

	def _output_dim_multiplier(self):
		return 2

	def _elementwise_forward(self, x, elementwise_params):
		assert elementwise_params.shape[-1] == self._output_dim_multiplier()
		unconstrained_scale,shift = self._unconstrained_scale_and_shift(elementwise_params)
		scale = self.scale_fn(unconstrained_scale)
		z = scale*x + shift
		ldj = sum_except_batch(torch.log(scale))

		return z,ldj

	def _elementwise_inverse(self, z, elementwise_params):
		assert elementwise_params.shape[-1] == self._output_dim_multiplier()
		unconstrained_scale,shift = self._unconstrained_scale_and_shift(elementwise_params)
		scale = self.scale_fn(unconstrained_scale)
		x = (z-shift)/scale
		return x

	def _unconstrained_scale_and_shift(self, elementwise_params):
		unconstrained_scale = elementwise_params[...,0]
		shift = elementwise_params[...,1]
		return unconstrained_scale,shift

















