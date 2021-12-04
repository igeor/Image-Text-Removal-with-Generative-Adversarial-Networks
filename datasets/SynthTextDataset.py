from torch.utils.data import Dataset, DataLoader
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch
from random import choice
from PIL import Image 
import os 

import os
import glob
import numpy as np
import random
import scipy.io
from torchvision.utils import save_image
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2 
import string 
import string_utils

pil_img = transforms.ToPILImage()

def load_dataset():
	mat = scipy.io.loadmat('/media/storage/datasets/SynthText/GT.mat')

	from collections import defaultdict
	groups = dict()

	print("SynthText Dataset contains: ", mat["imnames"][0].shape, "image names.", end=' ')

	cur_basename = ""
	for i, filename in enumerate(mat["imnames"][0]): # gia kathe filename
		filename = filename[0] 
		
		basename, extension = os.path.splitext(filename)

		image_base_name, image_class, image_no  = basename.split('_')
		basename = image_base_name +"_"+ image_class
		
		if basename != cur_basename:
			cur_basename = basename
			
		if cur_basename not in groups.keys():
			groups[cur_basename] = [filename]
		else:
			groups[cur_basename].append(filename)

	import pandas as pd 

	images_df = pd.DataFrame.from_dict(groups, orient='index')
	images_df["image_names"] = images_df.values.tolist()
	images_df = images_df["image_names"]

	
	"""
	datasets = list()

	for image_class, image_names in images_df.iteritems():
		dataset = list()
		for image_name in image_names:
			if image_name:
				dataset.append(os.path.join(ROOT, image_name))
		datasets.append(dataset)

	print("Number of clean images: ", len(datasets), datasets[0][0])
	"""
	return images_df


class SynthTextDataset(Dataset):

	def __init__(self, 
		transform=None,
		extra_text=False):

		self.ROOT = "/media/storage/datasets/SynthText/Images"
		self.transform = transform
		self.dataset_df = load_dataset()
		self.TARGET_ROOT = '/home/igeorvasilis/sdb/synth_text/clean_images'
		self. fonts = [cv2.FONT_HERSHEY_PLAIN,
			cv2.FONT_HERSHEY_COMPLEX_SMALL,
			cv2.FONT_HERSHEY_COMPLEX,
			cv2.FONT_HERSHEY_DUPLEX,
			cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
			cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
			cv2.FONT_HERSHEY_SIMPLEX,
			cv2.FONT_ITALIC]
		self.extra_text = extra_text

	def get_image_with_text(self, img, num_strings=10):
		img = np.asarray(img)
		#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		for i in range(random.randint(2,5)):
			font = random.choice(self.fonts)
			font_scale = random.randint(1,4)
			text = self.get_random_string(max_length=15)
			(text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
			text_offset_x = random.randint(10, 300)
			text_offset_y = random.randint(10, 400)
			box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
			col = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
			cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=col, thickness=random.randint(1,3))

		return img
	
	def get_random_string(self, max_length=20):
		
		length = random.randint(1, max_length)
		letters = string.ascii_lowercase
		result_str = ''.join(random.choice(letters) for i in range(length//2))
		digits = string.digits
		result_str += ''.join(random.choice(digits) for i in range(length//2))
		return string_utils.shuffle(result_str)


	def __getitem__(self, index):

		input_img_classname = self.dataset_df.keys()[index].split('/')[1]
		corrupted_images_namelist = self.dataset_df.iloc[index]

		while(True):
			source_image_name = choice(corrupted_images_namelist)
			target_image_name = choice(corrupted_images_namelist)
			if(source_image_name and target_image_name):
				break
		
		
		source_img_path = os.path.join(self.ROOT, source_image_name)
		target_img_path = os.path.join(self.TARGET_ROOT, input_img_classname) + ".png"
		#target_img_path = os.path.join(self.ROOT, target_image_name)
		
		input_image = Image.open(source_img_path).convert('RGB')
		target_image = Image.open(target_img_path).convert('RGB')
		if self.extra_text:
			input_image = self.get_image_with_text(input_image)

		if(self.transform):
			input_image = self.transform(input_image)
			target_image = self.transform(target_image)

		#input_img_classname = input_img_classname.split('/')[1]
		return target_image, input_image

	
	def __len__(self):
		return self.dataset_df.size