import cv2 # opencv
import numpy as np
import random 
import string 
import string_utils
from PIL import Image
from numpy import asarray

def get_random_string(max_length=20):
	length = random.randint(2, max_length)
	letters = string.ascii_lowercase
	result_str = ''.join(random.choice(letters) for i in range(length//2))
	digits = string.digits
	result_str += ''.join(random.choice(digits) for i in range(length//2))
	return string_utils.shuffle(result_str)


fonts = [cv2.FONT_HERSHEY_PLAIN,
		cv2.FONT_HERSHEY_COMPLEX_SMALL,
		cv2.FONT_HERSHEY_COMPLEX,
		cv2.FONT_HERSHEY_DUPLEX,
		cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
		cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
		cv2.FONT_HERSHEY_SIMPLEX,
		cv2.FONT_ITALIC]

def add_text(img):
	img = asarray(img)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	for i in range(random.randint(2,5)):
		font = random.choice(fonts)
		font_scale = random.randint(1,3)

		# set some text
		text = get_random_string(max_length=15)
		# get the width and height of the text box
		(text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
		# set the text start position   
		text_offset_x = random.randint(10, 300)
		text_offset_y = random.randint(10, 400)
		# make the coords of the box with a small padding of two pixels
		box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
		#cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
		col = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
		cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=col, thickness=random.randint(1,3))

	return img

