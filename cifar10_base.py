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





### cifar10 part

data = VectorizedCIFAR10()
train_loader, test_loader = data.get_data_loaders(4)
x = next(iter(train_loader))



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

def net(channels):
	return nn.Sequential(DenseNet(in_channels=channels//2,
		out_channels=channels,
		num_blocks=1,
		mid_channels=32,
		depth=2,
		growth=2,
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



model = Flow(base_dist=StandardNormal((12,16,16)),
			transforms=[
				UniformDequantization(num_bits=8),
				QuantizationBijection(),
				Squeeze2d(),
				AffineCouplingBijection(net(12)), ActNormBijection2d(12), Conv1x1(12),
				AffineCouplingBijection(net(12)), ActNormBijection2d(12), Conv1x1(12),
				AffineCouplingBijection(net(12)), ActNormBijection2d(12), Conv1x1(12),
				AffineCouplingBijection(net(12)), ActNormBijection2d(12), Conv1x1(12),
				AffineCouplingBijection(net(12)), ActNormBijection2d(12), Conv1x1(12),
				AffineCouplingBijection(net(12)), ActNormBijection2d(12), Conv1x1(12)]).to(device)



model = Flow(base_dist=StandardNormal((12,16,16)),
			transforms=[
				QuantizationBijection(),
				Squeeze2d(),
				AffineCouplingBijection(net(12)), ActNormBijection2d(12), Conv1x1(12),
				AffineCouplingBijection(net(12)), ActNormBijection2d(12), Conv1x1(12),
				AffineCouplingBijection(net(12)), ActNormBijection2d(12), Conv1x1(12),
				AffineCouplingBijection(net(12)), ActNormBijection2d(12), Conv1x1(12),
				AffineCouplingBijection(net(12)), ActNormBijection2d(12), Conv1x1(12),
				AffineCouplingBijection(net(12)), ActNormBijection2d(12), Conv1x1(12),
				AffineCouplingBijection(net(12)), ActNormBijection2d(12), Conv1x1(12),
				AffineCouplingBijection(net(12)), ActNormBijection2d(12), Conv1x1(12),
				AffineCouplingBijection(net(12)), ActNormBijection2d(12), Conv1x1(12),
				AffineCouplingBijection(net(12)), ActNormBijection2d(12), Conv1x1(12),
				AffineCouplingBijection(net(12)), ActNormBijection2d(12), Conv1x1(12),
				AffineCouplingBijection(net(12)), ActNormBijection2d(12), Conv1x1(12),
				AffineCouplingBijection(net(12)), ActNormBijection2d(12), Conv1x1(12),
				AffineCouplingBijection(net(12)), ActNormBijection2d(12), Conv1x1(12)]).to(device)

model = Flow(base_dist=StandardNormal((12,16,16)),
			transforms=[Squeeze2d(),
				AffineCouplingBijection(net(12)), ActNormBijection2d(12), Conv1x1(12)]).to(device)



x = imgTensor

transform = UniformDequantization(num_bits=8)
z,ldj = transform(x)

transform = Squeeze2d()
z,ldj = transform(z)

transform = AffineCouplingBijection(net(12))
z,ldj = transform(z)

transform = Conv1x1(12)
z,ldj = transform(z)

transform = ActNormBijection2d(12)
z,ldj = transform(z)

transform = AffineCouplingBijection(net(12))
z,ldj = transform(z)

transform = Conv1x1(12)
z,ldj = transform(z)

transform = ActNormBijection2d(12)
z,ldj = transform(z)

transform = AffineCouplingBijection(net(12))
z,ldj = transform(z)

transform = Conv1x1(12)
z,ldj = transform(z)

transform = ActNormBijection2d(12)
z,ldj = transform(z)






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

epoch = 0

optimizer = Adam(model.parameters(), lr=1e-4)

print('Training...')
for epoch in range(epoch,150):
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