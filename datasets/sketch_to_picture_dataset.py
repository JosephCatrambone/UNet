import os

import numpy
from PIL import Image
#from skimage.filters import difference_of_gaussians
from skimage.filters import sobel
from skimage.morphology import dilation, disk
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms


def sketch(numpy_image, marker_size:int = 2, lightness: float = None):
	"""Given a numpy matrix with pixels in the range [0,1] and order W,H,C create a fake 'drawing'.
	Assumes that the image MAY be multi-channel, but that all pixels are 0-1.
	"""
	assert numpy_image.min() >= 0 and numpy_image.max() <= 1
	# Make black and white using the wrong naive solution.
	if len(numpy_image.shape) > 2:
		img = numpy_image.sum(axis=-1) / numpy_image.shape[-1]
	else:
		img = numpy_image
	#filtered_image = difference_of_gaussians(img, 1, 4)  # One is a stonger edge here.
	filtered_image = sobel(img)  # 1.0 is the stronger edge here.
	# Go through and probabalistically add points.
	#noise = numpy.random.uniform(low=filtered_image.min()*(1.0-line_density), high=filtered_image.max()*line_density, size=filtered_image.shape[0:2])
	#drawing = (filtered_image < noise).astype(numpy.float)
	if lightness is None:
		lightness = 0.6
	drawing = (filtered_image > numpy.percentile(filtered_image, 100*lightness))

	if marker_size > 0:
		drawing = dilation(drawing, disk(marker_size))
	return 1.0 - drawing


class SketchToPictureDataset(Dataset):
	def __init__(self, base_image_folder, transform=None, target_width:int = 128, target_height:int = 128):
		super(SketchToPictureDataset, self).__init__()
		self.dir = base_image_folder
		self.resize_op = transforms.Resize(size=[target_height, target_width])
		self.target_width = target_width
		self.target_height = target_height
		# Normally we'd have another paired directory.
		self.transform = transform
		self.images = sorted(os.listdir(base_image_folder))

	def __len__(self):
		return len(self.images)

	def __getitem__(self, index):
		img_path = os.path.join(self.dir, self.images[index])
		img_pil = Image.open(img_path)

		# Maybe augment data:
		if self.transform == None:
			img = img_pil
		else:
			img = self.transform(img_pil)

		# Perhaps resize...
		if isinstance(img, torch.Tensor):
			img = self.resize_op(img)
		elif isinstance(img, Image.Image):
			img = img.resize((self.target_width, self.target_height))
			img = numpy.asarray(img.convert('RGB')) / 255.0
		else:
			pass  # This is a numpy array already.  Heck.

		# Now 'sketch' the image.
		drawing = sketch(img, marker_size=0)

		return torch.Tensor(drawing), torch.Tensor(img)