import os
import random
import string
from glob import iglob

import numpy
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset

class TextDetectionDataset(Dataset):
	def __init__(self, base_image_folder, font_directory, transform=None, target_width: int = 128, target_height: int = 128):
		super(TextDetectionDataset, self).__init__()
		self.dir = base_image_folder
		self.resize_op = transforms.Resize(size=[target_height, target_width])
		self.target_width = target_width
		self.target_height = target_height
		self.font_choices = list()
		for font_filename in iglob(font_directory):
			for font_size in [12, 14, 16, 24, 32]:
				self.font_choices.append(ImageFont.truetype(font_filename, font_size))
		# Normally we'd have another paired directory.

		#self.transform = transform
		if transform is not None:
			print("Sorry, transforms are not supported for this dataset.  We randomize inside the generator.")

		self.images = sorted(os.listdir(base_image_folder))
		self.image_center = (self.target_width//2, self.target_height//2)
		self.max_text_offset = (self.target_width//4, self.target_height//4)

	def __len__(self):
		return len(self.images)

	def random_text_image(self):
		"""Generate randomly oriented white text on a black background.  RGB image.  Can be used as a mask."""
		# TODO: This isn't a pretty thing, but it works.
		s = "".join(random.choice(string.ascii_letters + string.punctuation + " " * 4) for _ in range(50))
		text_image = Image.new("RGB", (self.target_width, self.target_height), "black")
		d = ImageDraw.Draw(text_image)
		# d.line(((0, 100), (200, 100)), "gray")
		# d.line(((100, 0), (100, 200)), "gray")
		# Note that this translates before rotating, so our offset might be a little weird.
		text_position = (
			self.image_center[0] + random.randint(-self.max_text_offset[0], self.max_text_offset[0]),
			self.image_center[1] + random.randint(-self.max_text_offset[1], self.max_text_offset[1]),
		)
		d.text(text_position, s, fill="white", anchor="mm", font=random.choice(self.font_choices))
		text_image = text_image.rotate(random.randint(0, 359))
		return text_image

	def __getitem__(self, index):
		# We assume we're starting with PIL images for everything AND that they have no text.
		# Perform updates via the transform stack.
		img_path = os.path.join(self.dir, self.images[index])
		img_pil = Image.open(img_path)

		# Randomly mutate the input image.
		if random.choice([False, True]):
			img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
		if random.choice([False, True]):
			img_pil = img_pil.transpose(Image.FLIP_TOP_BOTTOM)
		if random.choice([False, True]):
			img_pil = img_pil.rotate(random.randint(0,359))

		# If the image is big enough for a random crop, do it.
		if img_pil.size[0] > self.target_width and img_pil.size[1] > self.target_height:
			left = random.randint(0, img_pil.size[0]-1-self.target_width)
			top = random.randint(0, img_pil.size[1] - 1 - self.target_height)
			img_pil = img_pil.crop((left, top, left+self.target_width, top+self.target_height))
		else:
			img_pil = img_pil.resize((self.target_width, self.target_height))

		# Generate some random text:
		text_image_mask = self.random_text_image().convert('L')

		# Glorious hack to make a red mask:
		# red_channel = img_pil[0].point(lambda i: i < 100 and 255)

		# Draw the text image on top of our sample image.
		# if (red * 0.299 + green * 0.587 + blue * 0.114) > 186 use  # 000000 else use #ffffff
		total_color = [0, 0, 0]
		total_pixels = self.target_width*self.target_height
		for y in range(self.target_height):
			for x in range(self.target_width):
				px = img_pil.getpixel((x,y))
				total_color[0] += px[0]
				total_color[1] += px[1]
				total_color[2] += px[2]
		avg_r = total_color[0]//total_pixels
		avg_g = total_color[1]//total_pixels
		avg_b = total_color[2]//total_pixels

		# Default to light color...
		text_color = [random.randint(200, 255), random.randint(200, 255), random.randint(200, 255)]
		if avg_r*0.299 + avg_g*0.587 + avg_b * 0.114 > 186:
			# Unless our image is bright, in which case use dark color.
			text_color = [random.randint(0, 50), random.randint(0, 50), random.randint(0, 50)]
		# Make a rectangle of this color and paste it in with an image mask.
		text_color_block = Image.new("RGB", (self.target_width, self.target_height), text_color)
		result_image = img_pil.paste(text_color_block, (0,0), text_image_mask)

		return torch.Tensor(numpy.asarray(result_image, dtype=numpy.float) / 255.0), torch.Tensor(numpy.asarray(text_image_mask, dtype=numpy.float) / 255.0)