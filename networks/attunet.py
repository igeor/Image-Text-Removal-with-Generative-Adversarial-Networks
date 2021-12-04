from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from torchvision.utils import save_image

class conv_block(nn.Module):
	"""
	Convolution Block 
	"""
	def __init__(self, in_ch, out_ch):
		super(conv_block, self).__init__()
		
		self.conv = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True))

	def forward(self, x):

		x = self.conv(x)
		return x


class up_conv(nn.Module):
	"""
	Up Convolution Block
	"""
	def __init__(self, in_ch, out_ch):
		super(up_conv, self).__init__()
		self.up = nn.Sequential(
			nn.Upsample(scale_factor=2),
			nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		x = self.up(x)
		return x

	
class Attention_block(nn.Module):
	"""
	Attention Block
	"""

	def __init__(self, F_g, F_l, F_int):
		super(Attention_block, self).__init__()

		self.W_g = nn.Sequential(
			nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
			nn.BatchNorm2d(F_int)
		)

		self.W_x = nn.Sequential(
			nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
			nn.BatchNorm2d(F_int)
		)

		self.psi = nn.Sequential(
			nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
			nn.BatchNorm2d(1),
			nn.Sigmoid()
		)

		self.relu = nn.ReLU(inplace=True)

	def forward(self, g, x):
		print(g.shape, x.shape)
		g1 = self.W_g(g)
		x1 = self.W_x(x)
		print(g1.shape, x1.shape)
		psi = self.relu(g1 + x1)
		print(psi.shape)
		psi = self.psi(psi)
		print(psi.shape)
		out = x * psi
		print(psi.shape)#
		#if(psi.shape[3] == 256):
		#	save_image(psi, 'psi.png')
		#input()
		print(out.shape)
		return out

		
class AttU_Net(nn.Module):
	"""
	Attention Unet implementation
	Paper: https://arxiv.org/abs/1804.03999
	"""
	def __init__(self, img_ch=3, output_ch=1):
		super(AttU_Net, self).__init__()

		n1 = 64
		filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

		self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.Conv1 = conv_block(img_ch, 64)
		self.Conv2 = conv_block(64, 64)
		self.Conv3 = conv_block(64, 64)
		self.Conv4 = conv_block(64, 64)
		self.Conv5 = conv_block(64, 64)

		self.Up5 = up_conv(64, 64)
		self.Att5 = Attention_block(F_g=64, F_l=64, F_int=64)
		self.Up_conv5 = conv_block(128, 64)

		self.Up4 = up_conv(64, 64)
		self.Att4 = Attention_block(F_g=64, F_l=64, F_int=64)
		self.Up_conv4 = conv_block(128, 64)

		self.Up3 = up_conv(64, 64)
		self.Att3 = Attention_block(F_g=64, F_l=64, F_int=64)
		self.Up_conv3 = conv_block(128, 64)

		self.Up2 = up_conv(64, 64)
		self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
		self.Up_conv2 = conv_block(128, 64)

		self.Up1 = up_conv(64, 64)
		self.Att1 = Attention_block(F_g=64, F_l=64, F_int=32)
		self.Up_conv1 = conv_block(128, 64)

		self.Conv = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

		#self.active = torch.nn.Sigmoid()


	def forward(self, x):
		

		e1 = self.Conv1(x)

		e2 = self.Maxpool1(e1);print("block1(e1)->",e2.shape)
		e2 = self.Conv2(e2)

		e3 = self.Maxpool1(e2);print("block2(e2)->",e3.shape)
		e3 = self.Conv2(e3)

		e4 = self.Maxpool1(e3);print("block3(e3)->",e4.shape)
		e4 = self.Conv2(e4)

		e5 = self.Maxpool1(e4);print("block4(e4)->",e5.shape)
		e5 = self.Conv2(e5)

		e6 = self.Maxpool1(e5);print("block5(e5)->",e6.shape)
		e6 = self.Conv2(e6)
		
		d6 = self.Up5(e6);print("up5(e6)->",d6.shape)
		
		x5 = self.Att5(g=d6, x=e5);print("Att5(g=d6, x=e5)->",d6.shape,x5.shape)
		d6 = torch.cat((x5, d6), dim=1);print("d6.shape: ",d6.shape)
		d6 = self.Up_conv5(d6);print("Up_conv5(d6)->",d6.shape)

		d5 = self.Up4(d6);print("up4(e4)->",d5.shape)
		x4 = self.Att4(g=d5, x=e4);print("Att4(g=d5, x=e4)->",x4.shape)
		d5 = torch.cat((x4, d5), dim=1);print("d5.shape: ",d5.shape)
		d5 = self.Up_conv4(d5);print("Up_conv4(d5)->",d5.shape)

		d4 = self.Up3(d5);print("up3(d4)->",d4.shape)
		x3 = self.Att3(g=d4, x=e3);print("Att3(g=d4, x=e3)->",x3.shape)
		d4 = torch.cat((x3, d4), dim=1);print("d4.shape: ",d4.shape)
		d4 = self.Up_conv3(d4);print("Up_conv3(d4)->",d4.shape)

		d3 = self.Up2(d4);print("up2(d3)->",d3.shape)
		x2 = self.Att2(g=d3, x=e2);print("Att2(g=d3, x=e2)->",x2.shape)
		d3 = torch.cat((x2, d3), dim=1);print("d3.shape: ",d3.shape)
		d3 = self.Up_conv2(d3);print("Up_conv2(d3)->",d3.shape)

		d2 = self.Up1(d3);print("up2(d2)->",d2.shape)
		x1 = self.Att1(g=d2, x=e1);print("Att2(g=d2, x=e1)->",x1.shape)
		for feature in range(x1.shape[1]):
			save_image(x1[:,feature,:,:], str(feature)+'.png')
		d2 = torch.cat((x1, d2), dim=1);print("d2.shape: ",d2.shape)
		d2 = self.Up_conv1(d2);print("Up_conv2(d2)->",d2.shape)

		out = self.Conv(d2);print("Conv(d3)->",out.shape)

	  #  out = self.active(out)

		return out


class myAttU_Net(nn.Module):
	"""
	Attention Unet implementation
	Paper: https://arxiv.org/abs/1804.03999
	"""
	def __init__(self, img_ch=3, output_ch=1):
		super(myAttU_Net, self).__init__()

		n1 = 64
		filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]


		self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.Conv1 = conv_block(img_ch, filters[0])
		self.Conv2 = conv_block(filters[0], filters[1])
		self.Conv3 = conv_block(filters[1], filters[2])
		self.Conv4 = conv_block(filters[2], filters[3])
		self.Conv5 = conv_block(filters[3], filters[4])

		self.Up5 = up_conv(filters[4], filters[3])
		self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
		self.Up_conv5 = conv_block(filters[4], filters[3])

		self.Up4 = up_conv(filters[3], filters[2])
		self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
		self.Up_conv4 = conv_block(filters[3], filters[2])

		self.Up3 = up_conv(filters[2], filters[1])
		self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
		self.Up_conv3 = conv_block(filters[2], filters[1])

		self.Up2 = up_conv(filters[1], filters[0])
		self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
		self.Up_conv2 = conv_block(filters[1], filters[0])

		self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

		#self.active = torch.nn.Sigmoid()


	def forward(self, x):
		e1 = self.Conv1(x)
		e2 = self.Maxpool1(e1)
		e2 = self.Conv2(e2)
		e3 = self.Maxpool2(e2)
		e3 = self.Conv3(e3)
		e4 = self.Maxpool3(e3)
		e4 = self.Conv4(e4)
		e5 = self.Maxpool4(e4)
		e5 = self.Conv5(e5)

		#print(x5.shape)
		d5 = self.Up5(e5);print(d5.shape)
		#print(d5.shape)
		x4 = self.Att5(g=d5, x=0.1*e4);print(x4.shape)
		d5 = torch.cat((x4, d5), dim=1);print(d5.shape)
		d5 = self.Up_conv5(d5);print(d5.shape)

		d4 = self.Up4(d5)
		x3 = self.Att4(g=d4, x=e3)
		d4 = torch.cat((x3, d4), dim=1)
		d4 = self.Up_conv4(d4)

		d3 = self.Up3(d4)
		x2 = self.Att3(g=d3, x=e2)
		d3 = torch.cat((x2, d3), dim=1)
		d3 = self.Up_conv3(d3)

		d2 = self.Up2(d3)
		x1 = self.Att2(g=d2, x=e1)
		d2 = torch.cat((x1, d2), dim=1)
		d2 = self.Up_conv2(d2)

		out = self.Conv(d2)

	  #  out = self.active(out)

		return out