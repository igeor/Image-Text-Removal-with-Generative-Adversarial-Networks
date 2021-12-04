import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
import os
from skimage import io
import copy
import random
from random import randint
import matplotlib
from random import randint, shuffle, choice
import string_utils
from PIL import Image, ImageDraw, ImageFont, ImageFile
from pathlib import Path
import string
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2 # opencv
import numpy as np
import random 
import string 
import string_utils
from PIL import Image
from numpy import asarray


class KonIQDataset(Dataset):

	def __init__(self, 
		input_dir,
		mask_in=False,
		transform=None,
		shuffle=False):

		self.input_dir = input_dir
		self.transform = transform
		self.shuffle = shuffle
		self.init_fonts()
		self.dataset = self.make_dataset(self.input_dir)
		self.mask_in=mask_in
		self. fonts = [cv2.FONT_HERSHEY_PLAIN,
			cv2.FONT_HERSHEY_COMPLEX_SMALL,
			cv2.FONT_HERSHEY_COMPLEX,
			cv2.FONT_HERSHEY_DUPLEX,
			cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
			cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
			cv2.FONT_HERSHEY_SIMPLEX,
			cv2.FONT_ITALIC]

	def make_dataset(self, root):
		'''
		* Reads a directory with data.
		* Returns a dataset as a list of image names
		'''
		dataset = sorted(os.listdir(root))
		if(self.shuffle):
			dataset = random.sample(dataset, len(dataset))
		return dataset

	

	def get_image_with_text(self, img, num_strings=10):
		img = asarray(img)
		#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		for i in range(random.randint(2,5)):
			font = random.choice(self.fonts)
			font_scale = random.randint(1,3)
			text = self.get_random_string(max_length=15)
			(text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
			text_offset_x = random.randint(10, 300)
			text_offset_y = random.randint(10, 400)
			box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
			col = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
			cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=col, thickness=random.randint(1,3))

		return img


	"""
	def get_image_with_text(self, real_img, num_strings=10):
		cond_image = copy.deepcopy(real_img)
		draw = ImageDraw.Draw(cond_image)

		if self.mask_in:
			mask = Image.new('RGB', cond_image.size)
			mask_draw  = ImageDraw.Draw(mask)

		font = ImageFont.truetype(random.choice(self.system_fonts), randint(50,55))
		
		for i in range(randint(1, num_strings)):
			colors = [(0,100,100), (60, 100, 10), (240,100,10), (39,100,10), (271,81,8), (120,76,8), (26,85,5), (0,0,5), (255,255,255)]
			color = choice(colors)
			spot = (randint(0,real_img.size[0]-50), randint(0,real_img.size[1]-50))
			string = self.get_random_string()
			draw.text(spot, string, color, font=font)
			if self.mask_in:
				mask_draw.text(spot, string, (255,255,255), font=font)
		
		if self.mask_in:
			mask = mask.convert('L')
			return cond_image, mask

		return cond_image
		

	"""

	def init_fonts(self):
		#self.system_fonts = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
		#family_font = "Sans"
		#falimy_font_list = list(filter(lambda x: family_font in x, self.system_fonts))
		#self.system_fonts =  falimy_font_list
		self.system_fonts = ['/usr/share/fonts/truetype/dejavu/DejaVuSansMono-BoldOblique.ttf',
		'/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf',
		'/usr/share/fonts/truetype/liberation/LiberationSansNarrow-BoldItalic.ttf']


	def get_random_string(self, max_length=20):
		length = randint(1, max_length)
		letters = string.ascii_lowercase
		result_str = ''.join(random.choice(letters) for i in range(length//2))
		digits = string.digits
		result_str += ''.join(random.choice(digits) for i in range(length//2))
		return string_utils.shuffle(result_str)




	def __getitem__(self, index):
		img_name = self.dataset[index]
		input_img_path =  os.path.join(self.input_dir, img_name)#[:len(img_name)-1])
		real_image = Image.open(input_img_path).convert('RGB')
		if self.mask_in:
			condition_image, mask = self.get_image_with_text(real_image)
		else:
			condition_image = self.get_image_with_text(real_image)
		
		
		if(self.transform):
			real_image = self.transform(real_image)
			condition_image = self.transform(condition_image)
			if self.mask_in:
				self.mask_transform = torchvision.transforms.Compose([
					torchvision.transforms.ToTensor(),
					torchvision.transforms.Resize((condition_image.shape[1], condition_image.shape[2]))
				])
				mask = self.mask_transform(mask)

		if self.mask_in:
			condition_image = torch.cat([condition_image, mask], dim=0)

		return real_image, condition_image
			
	def __len__(self):
		return len(self.dataset)