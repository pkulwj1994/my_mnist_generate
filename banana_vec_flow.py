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

import seaborn as sns


device = 'cuda' if torch.cuda.is_available() else 'cpu'






### mnist part
DataSets = Crescent(train_samples=5000, test_samples=5000)
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

def coupling_net(input_dim):
	return MLP(int(input_dim/2), input_dim,hidden_units=[512,256],
                                activation='relu',
                                in_lambda=None)

def coupling_net(input_dim):
	return MLP(int(input_dim/2), input_dim,hidden_units=[128,64,32,16],
	                            activation='relu',
	                            in_lambda=None)

model = Flow(base_dist=StandardNormal((2,)),
			transforms=[
			AffineCouplingBijection1d(coupling_net(2)),
			SwitchBijection1d(),AffineCouplingBijection1d(coupling_net(2)),
			SwitchBijection1d(),AffineCouplingBijection1d(coupling_net(2)),
			SwitchBijection1d(),AffineCouplingBijection1d(coupling_net(2)),
			SwitchBijection1d(),AffineCouplingBijection1d(coupling_net(2))]).to(device)





transform = AffineCouplingBijection1d(coupling_net(2))
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



epoch = 0

optimizer = Adam(model.parameters(), lr=1e-7)

for epoch in range(epoch,400):
	l = 0.0
	for i,x in enumerate(train_loader):
		optimizer.zero_grad()
		loss = -model.log_prob(x.to(device)).mean()
		loss.backward()
		optimizer.step()
		l += loss.detach().cpu().item()
		for transform in model.transforms:
			transform.cuda()
			_,ldj = transform((x.cuda()))
			print('layer ldj: {}'.format(ldj.detach().cpu().mean().item()))
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


transform = model.transforms[0].cuda()
z,ldj = transform(x.cuda())
print(ldj.detach().cpu().mean().item())




vecs = next(iter(test_loader))[:128]
samples = model.sample(128).cpu()


fig = plt.figure()
sns.kdeplot(vecs[:,0],vecs[:,1],shade=True,label='data')
plt.legend()
sns.kdeplot(samples[:,0],samples[:,1],shade=True,label='model')
plt.legend()
scatter_fig = fig.get_figure()
scatter_fig.savefig('./banana_data_and_model_density.png', dpi = 400)


fig = plt.figure()
sns.kdeplot(vecs[:,0],vecs[:,1],shade=True)
scatter_fig = fig.get_figure()
scatter_fig.savefig('./banana_data_density.png', dpi = 400)

fig = plt.figure()
sns.kdeplot(samples[:,0],samples[:,1],shade=True)
scatter_fig = fig.get_figure()
scatter_fig.savefig('./banana_model_density.png', dpi = 400)

fig = plt.figure()
plt.scatter(vecs[:,0],vecs[:,1],s=5)
scatter_fig = fig.get_figure()
scatter_fig.savefig('./banana_data.png', dpi = 400)

fig = plt.figure()
plt.scatter(samples[:,0],samples[:,1],s=5)
scatter_fig = fig.get_figure()
scatter_fig.savefig('./banana_model.png', dpi = 400)

fig = plt.figure()
plt.scatter(vecs[:,0],vecs[:,1],s=1,label='data')
plt.legend()
plt.scatter(samples[:,0],samples[:,1],s=1,label='model')
plt.legend()
scatter_fig = fig.get_figure()
scatter_fig.savefig('./banana_data_and_model.png', dpi = 400)


