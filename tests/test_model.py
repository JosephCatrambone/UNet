import torch

from model import UNet

def test_sanity():
	batch_size = 5
	x = torch.randn((batch_size, 1, 200, 200))
	model = UNet(in_channels=1, out_channels=1)
	pred = model(x)
	assert x.shape == pred.shape