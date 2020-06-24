# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import torch
import numpy as np
import pdb
from utils import choose_nonlinearity
import torch.nn.functional as F
import math
class MLP(torch.nn.Module):
  '''Just a salt-of-the-earth MLP'''
  def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity='tanh'):
    super(MLP, self).__init__()
    self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
    self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear3 = torch.nn.Linear(hidden_dim, output_dim, bias=None)

    for l in [self.linear1, self.linear2, self.linear3]:
      torch.nn.init.orthogonal_(l.weight) # use a principled initialization

    self.nonlinearity = choose_nonlinearity(nonlinearity)

  def forward(self, x, separate_fields=False):
    h = self.nonlinearity( self.linear1(x) )
    h = self.nonlinearity( self.linear2(h) )
    return self.linear3(h)

class MLPAutoencoder(torch.nn.Module):
  '''A salt-of-the-earth MLP Autoencoder + some edgy res connections'''
  def __init__(self, input_dim, hidden_dim, latent_dim, nonlinearity='tanh'):
    super(MLPAutoencoder, self).__init__()
    self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
    self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear3 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear4 = torch.nn.Linear(hidden_dim, latent_dim)

    self.linear5 = torch.nn.Linear(latent_dim, hidden_dim)
    self.linear6 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear7 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear8 = torch.nn.Linear(hidden_dim, input_dim)

    for l in [self.linear1, self.linear2, self.linear3, self.linear4, \
              self.linear5, self.linear6, self.linear7, self.linear8]:
      torch.nn.init.orthogonal_(l.weight)  # use a principled initialization

    self.nonlinearity = choose_nonlinearity(nonlinearity)

  def encode(self, x):
    h = self.nonlinearity( self.linear1(x) )
    h = h + self.nonlinearity( self.linear2(h) )
    h = h + self.nonlinearity( self.linear3(h) )
    return self.linear4(h)

  def decode(self, z):
    h = self.nonlinearity( self.linear5(z) )
    h = h + self.nonlinearity( self.linear6(h) )
    h = h + self.nonlinearity( self.linear7(h) )
    return self.linear8(h)

  def forward(self, x):
    z = self.encode(x)
    x_hat = self.decode(z)
    return x_hat

class ConvAutoencoder(torch.nn.Module):
  '''A salt-of-the-earth MLP Autoencoder + some edgy res connections'''

  def __init__(self, input_dim, hidden_dim, latent_dim, nonlinearity='tanh'):
    super(ConvAutoencoder, self).__init__()
    self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)

    self.conv2 = torch.nn.Conv2d(16, 64, 3, padding=1)

    self.linear1 = torch.nn.Linear(int((input_dim / 3) * 4), hidden_dim)
    self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear3 = torch.nn.Linear(hidden_dim, latent_dim)

    self.linear4 = torch.nn.Linear(latent_dim, hidden_dim)
    self.linear5 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear6 = torch.nn.Linear(hidden_dim, int((input_dim / 3) * 4))

    self.t_conv1 = torch.nn.ConvTranspose2d(64, 16, kernel_size=2, stride=2)
    self.t_conv2 = torch.nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2)

    self.pool = torch.nn.MaxPool2d(2, 2)

    for l in [self.conv1, self.conv2, self.linear1, self.linear2, self.linear3, self.linear4, \
              self.linear5, self.linear6, self.t_conv1, self.t_conv2]:
      torch.nn.init.orthogonal_(l.weight)  # use a principled initialization

    self.nonlinearity = choose_nonlinearity(nonlinearity)

  def encode(self, x):
    self.side = int(math.sqrt(x.shape[1] / 6))
    self.batch = int(x.shape[0])
    k = x.view(x.shape[0], 2 * self.side, self.side, 3) # Assume image is RGB
    k = k.permute(0, 3, 1, 2)

    h = F.relu(self.conv1(k))
    h = self.pool(h)
    h = F.relu(self.conv2(h))
    h = self.pool(h)
    h = torch.flatten(h, start_dim=1)
    h = self.nonlinearity(self.linear1(h))
    h = h + self.nonlinearity(self.linear2(h))
    return self.nonlinearity(self.linear3(h))

  def decode(self, z):

    h = self.nonlinearity(self.linear4(z))
    h = h + self.nonlinearity(self.linear5(h))
    h = self.linear6(h)
    # pdb.set_trace()
    h = h.view(self.batch, 64, int(self.side/2), int(self.side/4))
    h = F.relu(self.t_conv1(h))
    h = F.relu(self.t_conv2(h))
    h = h.permute(0, 2, 3, 1)
    return torch.flatten(h, start_dim=1)

  def forward(self, x):
    z = self.encode(x)
    x_hat = self.decode(z)
    return x_hat