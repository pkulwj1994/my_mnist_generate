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


vutils.save_image(img.cpu().float(), fp='mnist_data.png', nrow=8)
vutils.save_image(samples.cpu().float(), fp='mnist_vae.png', nrow=8)