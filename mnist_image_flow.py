from activations import *
from datasets import *
from transforms import *
from utils import *

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






### mnist part
DataSets = RawMNIST(flatten=False)
train_loader, test_loader = DataSets.get_data_loaders(128)


imgTensor = next(iter(train_loader))
x = imgTensor




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


def net(channels):
	return nn.Sequential(DenseNet(in_channels=channels//2,
		out_channels=channels,
		num_blocks=1,
		mid_channels=16,
		depth=4,
		growth=4,
		dropout=0.0,
		gated_conv=True,
		zero_init=True),
	ElementwiseParams2d(2))


def coupling_net(input_dim):
	return MLP(int(input_dim/2), input_dim,hidden_units=[1024,512,256,128],
                                activation='relu',
                                in_lambda=None)

def coupling_net(input_dim):
	return MLP(int(input_dim/2), input_dim,hidden_units=[128,64],
	                            activation='relu',
	                            in_lambda=None)

model = Flow(base_dist=StandardNormal((784,)),
			transforms=[
			AffineCouplingBijection1d(coupling_net(784)),
			SwitchBijection1d(),AffineCouplingBijection1d(coupling_net(784)),
			SwitchBijection1d(),AffineCouplingBijection1d(coupling_net(784))]).to(device)


model = Flow(base_dist=StandardNormal((4,14,14)),
			transforms=[
				Squeeze2d(),
				AffineCouplingBijection(net(4)), ActNormBijection2d(4), Conv1x1(4),
				AffineCouplingBijection(net(4)), ActNormBijection2d(4), Conv1x1(4)]).to(device)



transform = AffineCouplingBijection1d(coupling_net(784))
z,ldj = transform(x)

transform = ActNormBijection(784)
z,ldj = transform(z)

transform = AffineCouplingBijection1d(coupling_net(784))
z,ldj = transform(z)

transform = ActNormBijection(784)
z,ldj = transform(z)



optimizer = Adam(model.parameters(), lr=1e-2)

l = 0.0
x = next(iter(train_loader))
optimizer.zero_grad()
loss = -model.log_prob(x.to(device)).mean()
loss.backward()
optimizer.step()
l += loss.detach().cpu().item()
print('Epoch: {}/{}, Iter: {}/{}, Nats: {:.3f}'.format(epoch+1,20,i+1,len(train_loader),l/(i+1)),end='\r')
print('')

print_params(model)

torch.save(model.state_dict(),'./saved_model/mnist_flow_single_affinecoupling_layer.pth')


model = Flow(base_dist=StandardNormal((784,)),
			transforms=[
			LogisticBijection1d(),
			AffineCouplingBijection1d(coupling_net(784))]).to(device)
model.load_state_dict(torch.load('./saved_model/mnist_flow_single_affinecoupling_layer.pth'))
model.eval()


torch.save(model, PATH)
model = torch.load(PATH)
model.eval()



base_dist = model.base_dist
transform = model.transforms[0].cuda()

x = x[2,:].cuda().unsqueeze(0)

x = x.unsqueeze(0)

id,x2 = torch.chunk(x.cuda(),2,dim=1)
elementwise_params = transform.coupling_net(id)
unconstrained_scale, shift = transform._unconstrained_scale_and_shift(elementwise_params)
unconstrained_scale = torch.clamp(unconstrained_scale,-6,6)
scale = torch.exp(unconstrained_scale)
scale*x2 + shift
torch.sum(unconstrained_scale,dim=1)




test_f = lambda x,a: -1/2*np.log(2*np.pi) - 1/2*(a*x)**2 + np.log(a)

lis = []
for x in np.linspace(0,1,100):
	for a in np.linspace(0,30,100):
		print(test_f(x,a))
		lis.append(test_f(x,a))


x = 0.5

a = 2

x = 0.5/np.exp(1)

lam = 1

np.log(lam) - lam*x


_,__ = transform((z.cuda()))
_,__ = torch.chunk(_,2,dim=1)

z,ldj = transform((x.cuda()))

base_dist.log_prob(z)

ldj -z.shape[1]/2*np.log(2*np.pi)-1/2*torch.sum(z**2,dim=1)



epoch = 0

optimizer = Adam(model.parameters(), lr=5e-6)

for epoch in range(epoch,2000):
	l = 0.0
	for i,x in enumerate(train_loader):
		optimizer.zero_grad()
		loss = -model.log_prob(x.to(device)).mean()
		loss.backward()
		optimizer.step()
		l += loss.detach().cpu().item()
		print('Epoch: {}/{}, Iter: {}/{}, Nats: {:.3f}'.format(epoch+1,20,i+1,len(train_loader),l/(i+1)),end='\r')

	with torch.no_grad():
		samples = model.sample(64)
		vutils.save_image(samples.view(samples.shape[0],1,28,28).cpu().float(), fp='mnist_flow.png', nrow=8)

	print('')



for epoch in range(epoch,400):
	l = 0.0
	for i,x in enumerate(train_loader):
		optimizer.zero_grad()
		loss = -model.log_prob(x.to(device)).mean()
		loss.backward()
		optimizer.step()
		l += loss.detach().cpu().item()

		z = x
		for transform in model.transforms:
			transform.cuda()
			if isinstance(transform, AffineCouplingBijection1d):
				id,z2 = torch.chunk(z,2,dim=1)
				elementwise_params = transform.coupling_net(z2)
				unconstrained_scale, shift = transform._unconstrained_scale_and_shift(elementwise_params)
				z,ldj = transform((z.cuda()))
				print('layer ldj: {}'.format(ldj.detach().cpu().mean().item()))
				print('layer unconstrained_scale: {}'.format(unconstrained_scale.detach().cpu().mean().item()))
				print('layer mean output: {}'.format(z.detach().cpu().mean().item()))
			else:
				z,ldj = transform((z.cuda()))
				print('layer ldj: {}'.format(ldj.detach().cpu().mean().item()))
				print('layer mean output: {}'.format(z.detach().cpu().mean().item()))

		print('######################################')
		print('Epoch: {}/{}, Iter: {}/{}, Nats: {:.3f}'.format(epoch+1,20,i+1,len(train_loader),l/(i+1)),end='\r')
	print('')



print('Testing...')
with torch.no_grad():
    l = 0.0
    for i, x in enumerate(test_loader):
        loss = -model.log_prob(x.to(device)).sum() / (math.log(2) * x.numel())
        l += loss.detach().cpu().item()
        print('Iter: {}/{}, Bits/dim: {:.3f}'.format(i+1, len(test_loader), l/(i+1)))
    print('')



img = next(iter(test_loader))[:64]
samples = model.sample(64)

import torchvision.utils as vutils


vutils.save_image(img.view(img.shape[0],1,28,28).cpu().float(), fp='mnist_data.png', nrow=8)
vutils.save_image(samples.view(samples.shape[0],1,28,28).cpu().float(), fp='mnist_flow.png', nrow=8)













