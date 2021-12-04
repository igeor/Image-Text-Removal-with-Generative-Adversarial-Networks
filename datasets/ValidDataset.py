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


class ValidDataset(Dataset):

	def __init__(self, 
		input_dir,
		mask_in=False,
		transform=None,
		shuffle=False):

		self.input_dir = input_dir
		self.transform = transform
		self.shuffle = shuffle
		self.dataset = self.make_dataset(self.input_dir)
		

	def make_dataset(self, root):
		'''
		* Reads a directory with data.
		* Returns a dataset as a list of image names
		'''
		dataset = sorted(os.listdir(root))
		if(self.shuffle):
			dataset = random.sample(dataset, len(dataset))
		return dataset

	

	def __getitem__(self, index):
		img_name = self.dataset[index]
		input_img_path =  os.path.join(self.input_dir, img_name)#[:len(img_name)-1])
		real_image = Image.open(input_img_path).convert('RGB')
		
		if(self.transform):
			real_image = self.transform(real_image)

		return real_image
			
	def __len__(self):
		return len(self.dataset)